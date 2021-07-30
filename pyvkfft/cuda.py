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
import pycuda.gpuarray as cua
import pycuda.driver as cu_drv
from .base import primes, calc_transform_axes

# np.complex32 does not exist yet https://github.com/numpy/numpy/issues/14753
complex64 = np.dtype([('re', np.float16), ('im', np.float16)])


def load_library(basename):
    if platform.system() == 'Windows':
        ext = '.dll'
    else:
        ext = sysconfig.get_config_var('SO')
    return ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__) or os.path.curdir, basename + ext))


_vkfft_cuda = load_library("_vkfft_cuda")


class _types:
    """Aliases"""
    vkfft_config = ctypes.c_void_p
    stream = ctypes.c_void_p
    vkfft_app = ctypes.c_void_p


_vkfft_cuda.make_config.restype = ctypes.c_void_p
_vkfft_cuda.make_config.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                                    ctypes.c_void_p, ctypes.c_void_p, _types.stream, ctypes.c_int,
                                    ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int]

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


class VkFFTApp:
    """
    VkFFT application interface, similar to a cuFFT plan.
    """

    def __init__(self, shape, dtype: type, ndim=None, inplace=True, stream=None, norm=1,
                 r2c=False, batch=1, buffer_allocator=None, **kwargs):
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
        :param stream: the stream to use for the FFT calculation. If None,
            the default one will be used
        :param norm: if 0, every transform multiplies the L2 norm of the array
            by its size (or the size of the transformed array if ndim<d.ndim).
            if 1 (the default), the inverse transform divides the L2 norm
            by the array size, so FFT+iFFT will keep the array norm.
            if "ortho", each transform will keep the L2 norm, but that will currently
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

        # Experimental parameters. Not much difference is seen, so don't document this,
        # VkFFT default parameters seem fine.
        if "disableReorderFourStep" in kwargs:
            self.disableReorderFourStep = kwargs["disableReorderFourStep"]
        else:
            self.disableReorderFourStep = -1
        if "registerBoost" in kwargs:
            self.registerBoost = kwargs["registerBoost"]
        else:
            self.registerBoost = -1

        self.stream = stream
        self.norm = norm

        # Reshape to collapse non-transformed axes if necessary (batch)
        self.batch_shape = None

        # Precision: number of bytes per
        if dtype in [np.float16, complex64]:
            self.precision = 2
        elif dtype in [np.float32, np.complex64]:
            self.precision = 4
        elif dtype in [np.float64, np.complex128]:
            self.precision = 8
        self.config = self._make_config()
        if self.config is None:
            raise RuntimeError("Error creating VkFFTConfiguration. Was the CUDA context properly initialised ?")
        self.app = _vkfft_cuda.init_app(self.config)
        if self.app is None:
            raise RuntimeError("Error creating VkFFTApplication. Was the CUDA driver initialised .")
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
        nx, ny, nz = self.shape[-1], 1, 1
        if len(self.shape) > 1:
            ny = self.shape[-2]

        if len(self.shape) > 2:
            nz = self.shape[-3]

        if len(self.shape) > 3 and self.ndim < 3:
            # Collapse axes for dimensions >=3 as VkFFT works on 3D arrays
            self.batch_shape = tuple([np.prod(self.shape[:-2])] + list(self.shape[-2:]))
            nz = self.batch_shape[0]

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
        if self.stream is None:
            s = 0
        else:
            s = self.stream.handle

        if self.norm == "ortho":
            norm = 0
        else:
            norm = self.norm

        # We pass fake buffer pointer addresses to VkFFT. The real ones will be
        # given when performing the actual FFT.
        if self.inplace:
            config = _vkfft_cuda.make_config(nx, ny, nz, self.ndim, 1, 0, s,
                                             norm, self.precision, int(self.r2c),
                                             int(self.disableReorderFourStep), int(self.registerBoost))
        else:
            config = _vkfft_cuda.make_config(nx, ny, nz, self.ndim, 1, 2, s,
                                             norm, self.precision, int(self.r2c),
                                             int(self.disableReorderFourStep), int(self.registerBoost))
        return config

    def fft(self, src: cua.GPUArray, dest: cua.GPUArray = None):
        """
        Compute the forward FFT
        :param src: the source GPUarray
        :param dest: the destination GPUarray. Should be None for an inplace transform
        :return: the transformed array. For a R2C inplace transform, the complex view of the
            array is returned.
        """
        if self.inplace:
            if dest is not None:
                if src.gpudata != dest.gpudata:
                    raise RuntimeError("VkFFTApp.fft: dest is not None but this is an inplace transform")
            if self.batch_shape is not None:
                s = src.reshape(self.batch_shape)
            else:
                s = src
            _vkfft_cuda.fft(self.app, int(s.gpudata), int(s.gpudata))
            if self.norm == "ortho":
                if self.precision == 2:
                    src *= np.float16(self._get_fft_scale(norm=0))
                elif self.precision == 4:
                    src *= np.float32(self._get_fft_scale(norm=0))
                elif self.precision == 8:
                    src *= np.float64(self._get_fft_scale(norm=0))
            if self.r2c:
                if src.dtype == np.float32:
                    return src.view(dtype=np.complex64)
                elif src.dtype == np.float64:
                    return src.view(dtype=np.complex128)
            return src
        else:
            if dest is None:
                raise RuntimeError("VkFFTApp.fft: dest is None but this is an out-of-place transform")
            elif src.gpudata == dest.gpudata:
                raise RuntimeError("VkFFTApp.fft: dest and src are identical but this is an out-of-place transform")
            if self.r2c:
                assert (src.size == dest.size // dest.shape[-1] * 2 * (dest.shape[-1] - 1))
            if self.batch_shape is not None:
                s = src.reshape(self.batch_shape)
                if self.r2c:
                    c_shape = tuple(list(self.batch_shape[:-1]) + [self.batch_shape[-1] // 2 + 1])
                    d = dest.reshape(c_shape)
                else:
                    d = dest.reshape(self.batch_shape)
            else:
                s, d = src, dest
            _vkfft_cuda.fft(self.app, int(s.gpudata), int(d.gpudata))
            if self.norm == "ortho":
                if self.precision == 2:
                    dest *= np.float16(self._get_fft_scale(norm=0))
                elif self.precision == 4:
                    dest *= np.float32(self._get_fft_scale(norm=0))
                elif self.precision == 8:
                    dest *= np.float64(self._get_fft_scale(norm=0))
            return dest

    def ifft(self, src: cua.GPUArray, dest: cua.GPUArray = None):
        """
        Compute the backward FFT
        :param src: the source GPUarray
        :param dest: the destination GPUarray. Should be None for an inplace transform
        :return: the transformed array. For a C2R inplace transform, the float view of the
            array is returned.
        """
        if self.inplace:
            if dest is not None:
                if src.gpudata != dest.gpudata:
                    raise RuntimeError("VkFFTApp.fft: dest!=src but this is an inplace transform")
            if self.batch_shape is not None:
                if self.r2c:
                    src_shape = tuple(list(self.batch_shape[:-1]) + [self.batch_shape[-1] // 2])
                    s = src.reshape(src_shape)
                else:
                    s = src.reshape(self.batch_shape)
            else:
                s = src
            _vkfft_cuda.ifft(self.app, int(s.gpudata), int(s.gpudata))
            if self.norm == "ortho":
                if self.precision == 2:
                    src *= np.float16(self._get_ifft_scale(norm=0))
                elif self.precision == 4:
                    src *= np.float32(self._get_ifft_scale(norm=0))
                elif self.precision == 8:
                    src *= np.float64(self._get_ifft_scale(norm=0))
            if self.r2c:
                if src.dtype == np.complex64:
                    return src.view(dtype=np.float32)
                elif src.dtype == np.complex128:
                    return src.view(dtype=np.float64)
            return src
        if not self.inplace:
            if dest is None:
                raise RuntimeError("VkFFTApp.ifft: dest is None but this is an out-of-place transform")
            elif src.gpudata == dest.gpudata:
                raise RuntimeError("VkFFTApp.ifft: dest and src are identical but this is an out-of-place transform")
            if self.r2c:
                assert (dest.size == src.size // src.shape[-1] * 2 * (src.shape[-1] - 1))
                # Special case, src and dest buffer sizes are different,
                # VkFFT is configured to go back to the source buffer
                if self.batch_shape is not None:
                    src_shape = tuple(list(self.batch_shape[:-1]) + [self.batch_shape[-1] // 2 + 1])
                    s = src.reshape(src_shape)
                    d = dest.reshape(self.batch_shape)
                else:
                    s, d = src, dest
                _vkfft_cuda.ifft(self.app, int(d.gpudata), int(s.gpudata))
            else:
                if self.batch_shape is not None:
                    s = src.reshape(self.batch_shape)
                    d = dest.reshape(self.batch_shape)
                else:
                    s, d = src, dest
                _vkfft_cuda.ifft(self.app, int(s.gpudata), int(d.gpudata))
            if self.norm == "ortho":
                if self.precision == 2:
                    dest *= np.float16(self._get_ifft_scale(norm=0))
                elif self.precision == 4:
                    dest *= np.float32(self._get_ifft_scale(norm=0))
                elif self.precision == 8:
                    dest *= np.float64(self._get_ifft_scale(norm=0))
            return dest

    def _get_fft_scale(self, norm):
        """Return the scale factor by which an array must be multiplied to keep its L2 norm
        after a forward FT
        :param norm: the norm option for which the scale is computed, either 0 or 1
        :return: the scale factor
        """
        s = np.sqrt(np.prod(self.shape[-self.ndim:]))
        if self.r2c and self.inplace:
            s *= np.sqrt((self.shape[-1] - 2) / self.shape[-1])
        if norm == 0 or norm == 1:
            return 1 / s
        elif norm == "ortho":
            return 1
        raise RuntimeError("Unknown norm choice !")

    def get_fft_scale(self):
        """Return the scale factor by which an array must be multiplied to keep its L2 norm
        after a forward FT
        """
        return self._get_fft_scale(self.norm)

    def _get_ifft_scale(self, norm):
        """Return the scale factor by which an array must be multiplied to keep its L2 norm
        after a backward FT
        :param norm: the norm option for which the scale is computed, either 0 or 1
        :return: the scale factor
        """
        s = np.sqrt(np.prod(self.shape[-self.ndim:]))
        if self.r2c and self.inplace:
            s *= np.sqrt((self.shape[-1] - 2) / self.shape[-1])
        if norm == 0:
            return 1 / s
        elif norm == 1:
            return s
        elif norm == "ortho":
            return 1
        raise RuntimeError("Unknown norm choice !")

    def get_ifft_scale(self):
        """Return the scale factor by which an array must be multiplied to keep its L2 norm
        after a backward FT
        """
        return self._get_ifft_scale(self.norm)
