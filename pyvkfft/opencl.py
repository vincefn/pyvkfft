# -*- coding: utf-8 -*-

# PyVkFFT
#   (c) 2021- : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import os
import platform
import warnings
import sysconfig
import ctypes
import numpy as np
import pyopencl as cl
import pyopencl.array as cla


def primes(n):
    """ Returns the prime decomposition of n as a list
    """
    v = [1]
    assert (n > 0)
    i = 2
    while i * i <= n:
        while n % i == 0:
            v.append(i)
            n //= i
        i += 1
    if n > 1:
        v.append(n)
    return v


# np.complex32 does not exist yet https://github.com/numpy/numpy/issues/14753
complex64 = np.dtype([('re', np.float16), ('im', np.float16)])


def load_library(basename):
    if platform.system() == 'Windows':
        ext = '.dll'
    else:
        ext = sysconfig.get_config_var('SO')
    return ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__) or os.path.curdir, basename + ext))


_vkfft_opencl = load_library("_vkfft_opencl")


class _types:
    """Aliases"""
    vkfft_config = ctypes.c_void_p
    vkfft_app = ctypes.c_void_p


_vkfft_opencl.make_config.restype = ctypes.c_void_p
_vkfft_opencl.make_config.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                      ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]

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


class VkFFTApp:
    """
    VkFFT application interface implementing a FFT plan.
    """

    def __init__(self, shape, dtype: type, queue: cl.CommandQueue, ndim=None, inplace=True, norm=1, r2c=False):
        """

        :param shape: the shape of the array to be transformed
        :param dtype: the numpy dtype of the source array (can be complex64 or complex128)
        :param queue: the pyopencl CommandQueue to use for the transform.
        :param ndim: the number of dimensions to use for the FFT. By default,
            uses the array dimensions. Can be smaller, e.g. ndim=2 for a 3D
            array to perform a batched 3D FFT on all the layers.
        :param inplace: if True (the default), performs an inplace transform and
            the destination array should not be given in fft() and ifft().
        :param norm: if 0, every transform multiplies the L2 norm of the array
            by its size (or the size of the transformed array if ndim<d.ndim).
            if 1 (the default), the inverse transform divides the L2 norm
            by the array size, so FFT+iFFT will keep the array norm.
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
        :raises RuntimeError: if the initialisation fails, e.g. if the CUDA
            driver has not been properly initialised.
        """
        self.shape = shape
        if ndim is None:
            self.ndim = len(shape)
        else:
            self.ndim = ndim
        self.inplace = inplace
        self.r2c = r2c
        self.queue = queue
        self.norm = norm
        # Precision: number of bytes per
        if dtype in [np.float16, complex64]:
            self.precision = 2
        elif dtype in [np.float32, np.complex64]:
            self.precision = 4
        elif dtype in [np.float64, np.complex128]:
            self.precision = 8
        self.config = self._make_config()
        if self.config is None:
            raise RuntimeError("Error creating VkFFTConfiguration. Was the OpenCL context properly initialised ?")
        self.app = _vkfft_opencl.init_app(self.config, queue.int_ptr)
        if self.app is None:
            raise RuntimeError("Error creating VkFFTApplication. Was the OpenCL context properly initialised ?")

    def __del__(self):
        """ Takes care of deleting allocated memory in the underlying
        VkFFTApplication and VkFFTConfiguration.
        """
        _vkfft_opencl.free_app(self.app)
        _vkfft_opencl.free_config(self.config)

    def _make_config(self):
        """ Create a vkfft configuration for a FFT transform"""
        nx, ny, nz = 1, 1, 1
        if len(self.shape) == 3:
            nz, ny, nx = self.shape
        elif len(self.shape) == 2:
            ny, nx = self.shape
        elif len(self.shape) == 1:
            nx = self.shape[0]
        if self.r2c:
            if self.inplace:
                # the last two columns are ignored in the R array, and will be used
                # in the C array with a size nx//2+1
                nx -= 2
            else:
                # raise RuntimeError("VkFFTApp: out-of-place R2C transform is not supported")
                pass
        if max(primes(nx)) > 13 or (max(primes(ny)) > 13 and self.ndim >= 2) \
                or (self.ndim >= 3 and max(primes(nz)) > 13):
            raise RuntimeError("The prime numbers of the FFT size is larger than 13")
        # We pass fake buffer pointer addresses to VkFFT. The real ones will be
        # given when performing the actual FFT.
        ctx = self.queue.context
        device = ctx.devices[0]
        platform = device.platform
        if self.inplace:
            config = _vkfft_opencl.make_config(nx, ny, nz, self.ndim, 1, 0, platform.int_ptr,
                                               device.int_ptr, ctx.int_ptr,
                                               self.norm, self.precision, int(self.r2c))
        else:
            config = _vkfft_opencl.make_config(nx, ny, nz, self.ndim, 1, 2, platform.int_ptr,
                                               device.int_ptr, ctx.int_ptr,
                                               self.norm, self.precision, int(self.r2c))
        return config

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
                assert (src.size == dest.size // dest.shape[-1] * 2 * (dest.shape[-1] - 1))
            _vkfft_opencl.fft(self.app, int(src.data.int_ptr), int(dest.data.int_ptr), int(self.queue.int_ptr))
            return dest

    def ifft(self, src: cla.Array, dest: cla.Array = None):
        """
        Compute the backward FFT
        :param src: the source GPUarray
        :param dest: the destination GPUarray. Should be None for an inplace transform
        :return: the transformed array. For a C2R inplace transform, the float view of the
            array is returned.
        """
        if self.inplace:
            if dest is not None:
                if src.data.int_ptr != dest.data.int_ptr:
                    raise RuntimeError("VkFFTApp.fft: dest!=src but this is an inplace transform")
            _vkfft_opencl.ifft(self.app, int(src.data.int_ptr), int(src.data.int_ptr), int(self.queue.int_ptr))
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
                assert (dest.size == src.size // src.shape[-1] * 2 * (src.shape[-1] - 1))
                # Special case, src and dest buffer sizes are different,
                # VkFFT is configured to go back to the source buffer
                _vkfft_opencl.ifft(self.app, int(dest.data.int_ptr), int(src.data.int_ptr), int(self.queue.int_ptr))
            else:
                _vkfft_opencl.ifft(self.app, int(src.data.int_ptr), int(dest.data.int_ptr), int(self.queue.int_ptr))
            return dest
