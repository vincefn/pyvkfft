# -*- coding: utf-8 -*-

# PyVkFFT
#   (c) 2021- : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import ctypes
import numpy as np

try:
    import pycuda.driver as cu_drv
    has_pycuda = True
except ImportError:
    has_pycuda = False
try:
    import cupy as cp
    has_cupy = True
except ImportError:
    has_cupy = False
    if has_pycuda is False:
        raise ImportError("You need either PyCUDA or CuPy to use pyvkfft.cuda.")

from .base import load_library, primes, VkFFTApp as VkFFTAppBase

_vkfft_cuda = load_library("_vkfft_cuda")


class _types:
    """Aliases"""
    vkfft_config = ctypes.c_void_p
    stream = ctypes.c_void_p
    vkfft_app = ctypes.c_void_p


_vkfft_cuda.make_config.restype = ctypes.c_void_p
_vkfft_cuda.make_config.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                                    ctypes.c_void_p, ctypes.c_void_p, _types.stream, ctypes.c_int,
                                    ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_size_t,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int]

_vkfft_cuda.init_app.restype = ctypes.c_void_p
_vkfft_cuda.init_app.argtypes = [_types.vkfft_config]

_vkfft_cuda.fft.restype = None
_vkfft_cuda.fft.argtypes = [_types.vkfft_app, ctypes.c_void_p, ctypes.c_void_p]

_vkfft_cuda.ifft.restype = None
_vkfft_cuda.ifft.argtypes = [_types.vkfft_app, ctypes.c_void_p, ctypes.c_void_p]

_vkfft_cuda.free_app.restype = None
_vkfft_cuda.free_app.argtypes = [_types.vkfft_app]

_vkfft_cuda.free_config.restype = None
_vkfft_cuda.free_config.argtypes = [_types.vkfft_config]


class VkFFTApp(VkFFTAppBase):
    """
    VkFFT application interface, similar to a cuFFT plan.
    """

    def __init__(self, shape, dtype: type, ndim=None, inplace=True, stream=None, norm=1,
                 r2c=False, dct=False, axes=None, **kwargs):
        """

        :param shape: the shape of the array to be transformed. The number
            of dimensions of the array can be larger than the FFT dimensions,
            but only for 1D and 2D transforms. 3D FFT transforms can only
            be done on 3D arrays.
        :param dtype: the numpy dtype of the source array (can be complex64 or complex128)
        :param ndim: the number of dimensions to use for the FFT. By default,
            uses the array dimensions. Can be smaller, e.g. ndim=2 for a 3D
            array to perform a batched 3D FFT on all the layers. The FFT
            is always performed along the last axes if the array's number
            of dimension is larger than ndim, i.e. on the x-axis for ndim=1,
            on the x and y axes for ndim=2.
        :param inplace: if True (the default), performs an inplace transform and
            the destination array should not be given in fft() and ifft().
        :param stream: the pycuda.driver.Stream or cupy.cuda.Stream to use
            for the transform. If None, the default one will be used
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
        :raises RuntimeError: if the initialisation fails, e.g. if the CUDA
            driver has not been properly initialised.
        """
        super().__init__(shape, dtype, ndim=ndim, inplace=inplace, norm=norm, r2c=r2c, dct=dct, axes=axes, **kwargs)

        self.stream = stream

        self.config = self._make_config()
        if self.config is None:
            raise RuntimeError("Error creating VkFFTConfiguration. Was the CUDA context properly initialised ?")
        self.app = _vkfft_cuda.init_app(self.config)
        if self.app is None:
            raise RuntimeError("Error creating VkFFTApplication. Was the CUDA driver initialised ?")
        if has_pycuda:
            # TODO: This is a kludge to keep a reference to the context, so that it is deleted
            #  after the app in __delete__, which throws an error if the context does not exist
            #  anymore. Except that we cannot be sure this is the right context, if a stream
            #  has been given because we don't have access to cuStreamGetCtx from python...
            self._ctx = cu_drv.Context.get_current()

    def __del__(self):
        """ Takes care of deleting allocated memory in the underlying
        VkFFTApplication and VkFFTConfiguration.
        """
        _vkfft_cuda.free_app(self.app)
        _vkfft_cuda.free_config(self.config)

    def _make_config(self):
        """ Create a vkfft configuration for a FFT transform"""
        nx, ny, nz, n_batch = self.shape
        skipx, skipy, skipz = self.skip_axis
        if self.r2c and self.inplace:
            # the last two columns are ignored in the R array, and will be used
            # in the C array with a size nx//2+1
            nx -= 2

        s = 0
        if self.stream is not None:
            if has_pycuda:
                if isinstance(self.stream, cu_drv.Stream):
                    s = self.stream.handle
            if has_cupy:
                if isinstance(self.stream, cp.cuda.Stream):
                    s = self.stream.ptr

        if self.norm == "ortho":
            norm = 0
        else:
            norm = self.norm

        # We pass fake buffer pointer addresses to VkFFT. The real ones will be
        # given when performing the actual FFT.
        dest_gpudata = 2
        if self.inplace:
            dest_gpudata = 0

        return _vkfft_cuda.make_config(nx, ny, nz, self.ndim, 1, dest_gpudata, s,
                                       norm, self.precision, int(self.r2c), int(self.dct),
                                       int(self.disableReorderFourStep), int(self.registerBoost),
                                       int(self.use_lut), int(self.keepShaderCode),
                                       n_batch, skipx, skipy, skipz)

    def fft(self, src, dest=None):
        """
        Compute the forward FFT
        :param src: the source pycuda.gpuarray.GPUArray or cupy.ndarray
        :param dest: the destination GPU array. Should be None for an inplace transform
        :return: the transformed array. For a R2C inplace transform, the complex view of the
            array is returned.
        """
        use_cupy = False
        if has_cupy:
            if isinstance(src, cp.ndarray):
                use_cupy = True
        if use_cupy:
            src_ptr = src.__cuda_array_interface__['data'][0]
        else:
            src_ptr = src.gpudata
        if dest is not None:
            if use_cupy:
                dest_ptr = dest.__cuda_array_interface__['data'][0]
            else:
                dest_ptr = dest.gpudata
        else:
            dest_ptr = src_ptr
        if self.inplace:
            if src_ptr != dest_ptr:
                raise RuntimeError("VkFFTApp.fft: dest is not None but this is an inplace transform")
            _vkfft_cuda.fft(self.app, int(src_ptr), int(src_ptr))
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
            if src_ptr == dest_ptr:
                raise RuntimeError("VkFFTApp.fft: dest and src are identical but this is an out-of-place transform")
            if self.r2c:
                assert (dest.size == src.size // src.shape[-1] * (src.shape[-1] // 2 + 1))
            _vkfft_cuda.fft(self.app, int(src_ptr), int(dest_ptr))
            if self.norm == "ortho":
                dest *= self._get_fft_scale(norm=0)
            return dest

    def ifft(self, src, dest=None):
        """
        Compute the backward FFT
        :param src: the source pycuda.gpuarray.GPUArray or cupy.ndarray
        :param dest: the destination GPU array. Should be None for an inplace transform
        :return: the transformed array. For a C2R inplace transform, the float view of the
            array is returned.
        """
        use_cupy = False
        if has_cupy:
            if isinstance(src, cp.ndarray):
                use_cupy = True
        if use_cupy:
            src_ptr = src.__cuda_array_interface__['data'][0]
        else:
            src_ptr = src.gpudata
        if dest is not None:
            if use_cupy:
                dest_ptr = dest.__cuda_array_interface__['data'][0]
            else:
                dest_ptr = dest.gpudata
        else:
            dest_ptr = src_ptr
        if self.inplace:
            if dest is not None:
                if src_ptr != dest_ptr:
                    raise RuntimeError("VkFFTApp.fft: dest!=src but this is an inplace transform")
            _vkfft_cuda.ifft(self.app, int(src_ptr), int(src_ptr))
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
            if src_ptr == dest_ptr:
                raise RuntimeError("VkFFTApp.ifft: dest and src are identical but this is an out-of-place transform")
            if self.r2c:
                assert (src.size == dest.size // dest.shape[-1] * (dest.shape[-1] // 2 + 1))
                # Special case, src and dest buffer sizes are different,
                # VkFFT is configured to go back to the source buffer
                _vkfft_cuda.ifft(self.app, int(dest_ptr), int(src_ptr))
            else:
                _vkfft_cuda.ifft(self.app, int(src_ptr), int(dest_ptr))
            if self.norm == "ortho":
                dest *= self._get_ifft_scale(norm=0)
            return dest


def vkfft_version():
    """
    Get VkFFT version
    :return: version as X.Y.Z
    """
    int_ver = _vkfft_cuda.vkfft_version()
    return "%d.%d.%d" % (int_ver // 10000, (int_ver % 10000) // 100, int_ver % 100)
