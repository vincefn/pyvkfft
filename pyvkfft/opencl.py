# -*- coding: utf-8 -*-

# PyVkFFT
#   (c) 2021- : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import warnings
import ctypes
import numpy as np
import pyopencl as cl
import pyopencl.array as cla
from .base import load_library, primes, VkFFTApp as VkFFTAppBase

_vkfft_opencl = load_library("_vkfft_opencl")


class _types:
    """Aliases"""
    vkfft_config = ctypes.c_void_p
    vkfft_app = ctypes.c_void_p


_vkfft_opencl.make_config.restype = ctypes.c_void_p
_vkfft_opencl.make_config.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                                      ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                      ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_int,
                                      ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int]

_vkfft_opencl.init_app.restype = ctypes.c_void_p
_vkfft_opencl.init_app.argtypes = [_types.vkfft_config, ctypes.c_void_p]

_vkfft_opencl.fft.restype = None
_vkfft_opencl.fft.argtypes = [_types.vkfft_app, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

_vkfft_opencl.ifft.restype = None
_vkfft_opencl.ifft.argtypes = [_types.vkfft_app, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

_vkfft_opencl.free_app.restype = None
_vkfft_opencl.free_app.argtypes = [_types.vkfft_app]

_vkfft_opencl.free_config.restype = None
_vkfft_opencl.free_config.argtypes = [_types.vkfft_config]

_vkfft_opencl.vkfft_version.restype = ctypes.c_uint32
_vkfft_opencl.vkfft_version.argtypes = None


class VkFFTApp(VkFFTAppBase):
    """
    VkFFT application interface implementing a FFT plan.
    """

    def __init__(self, shape, dtype: type, queue: cl.CommandQueue, ndim=None, inplace=True, norm=1,
                 r2c=False, dct=False, axes=None, **kwargs):
        """
        Init function for the VkFFT application.

        :param shape: the shape of the array to be transformed. The number
            of dimensions of the array can be larger than the FFT dimensions.
        :param dtype: the numpy dtype of the source array (can be complex64 or complex128)
        :param queue: the pyopencl CommandQueue to use for the transform.
        :param ndim: the number of dimensions to use for the FFT. By default,
            uses the array dimensions. Can be smaller, e.g. ndim=2 for a 3D
            array to perform a batched 3D FFT on all the layers. The FFT
            is always performed along the last axes if the array's number
            of dimension is larger than ndim, i.e. on the x-axis for ndim=1,
            on the x and y axes for ndim=2, etc.. Unless axes are given.
        :param inplace: if True (the default), performs an inplace transform and
            the destination array should not be given in fft() and ifft().
        :param norm: if 0 (unnormalised), every transform multiplies the L2
            norm of the array by its size (or the size of the transformed
            array if ndim<d.ndim).
            if 1 (the default) or "backward", the inverse transform divides
            the L2 norm by the array size, so FFT+iFFT will keep the array norm.
            if "ortho", each transform will keep the L2 norm, but that will
            involve an extra read & write operation.
        :param r2c: if True, will perform a real->complex transform, where the
            complex destination is a half-hermitian array.
            For an inplace transform, if the input data shape is (...,nx), the input
            float array should have a shape of (..., nx+2), the last two columns
            being ignored in the input data, and the resulting
            complex array (using pycuda's GPUArray.view(dtype=np.complex64) to
            reinterpret the type) will have a shape (..., nx//2 + 1).
            For an out-of-place transform, if the input (real) shape is (..., nx),
            the output (complex) shape should be (..., nx//2+1).
            Note that for C2R transforms with ndim>=2, the source (complex) array
            is modified.
        :param dct: used to perform a Direct Cosine Transform (DCT) aka a R2R transform.
            An integer can be given to specify the type of DCT (1, 2, 3 or 4).
            if dct=True, the DCT type 2 will be performed, following scipy's convention.
        :param axes: a list or tuple of axes along which the transform should be made.
            if None, the transform is done along the ndim fastest axes, or all
            axes if ndim is None. Not allowed for R2C transforms
        :raises RuntimeError: if the initialisation fails, e.g. if the GPU
            driver has not been properly initialised, or if the transform dimensions
            are not allowed by VkFFT.
        """
        if dct == 4:
            if ndim is not None:
                if ndim > 1:
                    raise RuntimeError("DCT type IV is not supported for OpenCL for ndim>1")
            elif len(shape) > 1:
                raise RuntimeError("DCT type IV is not supported for OpenCL for ndim>1")
        super().__init__(shape, dtype, ndim=ndim, inplace=inplace, norm=norm, r2c=r2c, dct=dct, axes=axes, **kwargs)

        self.queue = queue

        if self.precision == 2 and 'cl_khr_fp16' not in self.queue.device.extensions:
            raise RuntimeError("Half precision required but cl_khr_fp16 extension is not available")
        if self.precision == 8 and 'cl_khr_fp64' not in self.queue.device.extensions:
            raise RuntimeError("Double precision required but cl_khr_fp64 extension is not available")

        self.config = self._make_config()

        if self.config is None:
            print("VkFFTApp:", shape, axes, ndim, r2c, "->", self.shape, self.skip_axis, self.ndim)
            raise RuntimeError("Error creating VkFFTConfiguration. Was the OpenCL context properly initialised ?")
        self.app = _vkfft_opencl.init_app(self.config, queue.int_ptr)
        if self.app is None:
            print("VkFFTApp:", shape, axes, ndim, r2c, "->", self.shape, self.skip_axis, self.ndim)
            raise RuntimeError("Error creating VkFFTApplication. Was the OpenCL context properly initialised ?")

    def __del__(self):
        """ Takes care of deleting allocated memory in the underlying
        VkFFTApplication and VkFFTConfiguration.
        """
        _vkfft_opencl.free_app(self.app)
        _vkfft_opencl.free_config(self.config)

    def _make_config(self):
        """ Create a vkfft configuration for a FFT transform"""
        nx, ny, nz, n_batch = self.shape
        skipx, skipy, skipz = self.skip_axis
        if self.r2c and self.inplace:
            # the last two columns are ignored in the R array, and will be used
            # in the C array with a size nx//2+1
            nx -= 2

        if self.norm == "ortho":
            norm = 0
        else:
            norm = self.norm

        # We pass fake buffer pointer addresses to VkFFT. The real ones will be
        # given when performing the actual FFT.
        ctx = self.queue.context
        device = ctx.devices[0]
        platform = device.platform
        dest_gpudata = 2
        if self.inplace:
            dest_gpudata = 0

        return _vkfft_opencl.make_config(nx, ny, nz, self.ndim, 1, dest_gpudata, platform.int_ptr,
                                         device.int_ptr, ctx.int_ptr,
                                         norm, self.precision, int(self.r2c), int(self.dct),
                                         int(self.disableReorderFourStep), int(self.registerBoost),
                                         int(self.use_lut), int(self.keepShaderCode),
                                         n_batch, skipx, skipy, skipz)

    def fft(self, src: cla.Array, dest: cla.Array = None):
        """
        Compute the forward FFT
        :param src: the source pyopencl Array
        :param dest: the destination pyopencl Array. Should be None for an inplace transform
        :return: the transformed array. For a R2C inplace transform, the complex view of the
            array is returned.
        """
        if self.inplace:
            if dest is not None:
                if src.data.int_ptr != dest.data.int_ptr:
                    raise RuntimeError("VkFFTApp.fft: dest is not None but this is an inplace transform")
            _vkfft_opencl.fft(self.app, int(src.data.int_ptr), int(src.data.int_ptr), int(self.queue.int_ptr))
            if self.norm == "ortho":
                src *= self._get_fft_scale(norm=0)
            if self.r2c:
                if src.dtype == np.float32:
                    return src.view(dtype=np.complex64)
                elif src.dtype == np.float64:
                    return src.view(dtype=np.complex128)
            return src
        else:
            if dest is None:
                raise RuntimeError("VkFFTApp.fft: dest is None but this is an out-of-place transform")
            elif src.data.int_ptr == dest.data.int_ptr:
                raise RuntimeError("VkFFTApp.fft: dest and src are identical but this is an out-of-place transform")
            if self.r2c:
                assert (dest.size == src.size // src.shape[-1] * (src.shape[-1] // 2 + 1))
            _vkfft_opencl.fft(self.app, int(src.data.int_ptr), int(dest.data.int_ptr), int(self.queue.int_ptr))
            if self.norm == "ortho":
                dest *= self._get_fft_scale(norm=0)
            return dest

    def ifft(self, src: cla.Array, dest: cla.Array = None):
        """
        Compute the backward FFT
        :param src: the source pyopencl.Array
        :param dest: the destination pyopencl.Array. Can be None for an inplace transform
        :return: the transformed array. For a C2R inplace transform, the float view of the
            array is returned.
        """
        if self.inplace:
            if dest is not None:
                if src.data.int_ptr != dest.data.int_ptr:
                    raise RuntimeError("VkFFTApp.fft: dest!=src but this is an inplace transform")
            _vkfft_opencl.ifft(self.app, int(src.data.int_ptr), int(src.data.int_ptr), int(self.queue.int_ptr))
            if self.norm == "ortho":
                src *= self._get_ifft_scale(norm=0)
            if self.r2c:
                if src.dtype == np.complex64:
                    return src.view(dtype=np.float32)
                elif src.dtype == np.complex128:
                    return src.view(dtype=np.float64)
            return src
        if not self.inplace:
            if dest is None:
                raise RuntimeError("VkFFTApp.ifft: dest is None but this is an out-of-place transform")
            elif src.data.int_ptr == dest.data.int_ptr:
                raise RuntimeError("VkFFTApp.ifft: dest and src are identical but this is an out-of-place transform")
            if self.r2c:
                assert (src.size == dest.size // dest.shape[-1] * (dest.shape[-1] // 2 + 1))
                # Special case, src and dest buffer sizes are different,
                # VkFFT is configured to go back to the source buffer
                _vkfft_opencl.ifft(self.app, int(dest.data.int_ptr), int(src.data.int_ptr),
                                   int(self.queue.int_ptr))
            else:
                _vkfft_opencl.ifft(self.app, int(src.data.int_ptr), int(dest.data.int_ptr), int(self.queue.int_ptr))
            if self.norm == "ortho":
                dest *= self._get_ifft_scale(norm=0)
            return dest


def vkfft_version():
    """
    Get VkFFT version
    :return: version as X.Y.Z
    """
    int_ver = _vkfft_opencl.vkfft_version()
    return "%d.%d.%d" % (int_ver // 10000, (int_ver % 10000) // 100, int_ver % 100)
