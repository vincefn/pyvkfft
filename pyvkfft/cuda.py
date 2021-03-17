# -*- coding: utf-8 -*-

# PyVkFFT
#   (c) 2021- : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import os
import platform
import sysconfig
import ctypes
import pycuda.gpuarray as cua


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
_vkfft_cuda.make_config.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_void_p]

_vkfft_cuda.init_app.restype = ctypes.c_void_p
_vkfft_cuda.init_app.argtypes = [_types.vkfft_config]

_vkfft_cuda.test_vkfft_cuda.restype = ctypes.c_int
_vkfft_cuda.test_vkfft_cuda.argtypes = [ctypes.c_int]

_vkfft_cuda.fft.restype = None
_vkfft_cuda.fft.argtypes = [_types.vkfft_app]

_vkfft_cuda.ifft.restype = None
_vkfft_cuda.ifft.argtypes = [_types.vkfft_app]

_vkfft_cuda.free_app.restype = None
_vkfft_cuda.free_app.argtypes = [_types.vkfft_app]

_vkfft_cuda.free_config.restype = None
_vkfft_cuda.free_config.argtypes = [_types.vkfft_config]


class VkFFTApp:
    """
    VkFFT application interface, similar to a cuFFT plan.
    """
    def __init__(self, d: cua.GPUArray, ndim=None):
        """

        :param d: the GPUArray for which the transform will be calculated
        :param ndim: the number of dimensions to use for the FFT. By default,
            uses the array dimensions. Can be smaller, e.g. ndim=2 for a 3D
            array to perform a batched 3D FFT on all the layers.
        :raises RuntimeError: if the initialisation fails, e.g. if the CUDA
            driver has not been properly initialised.
        """
        self.d = d
        if ndim is None:
            self.ndim = self.d.ndim
        else:
            self.ndim = ndim
        self.config = self._make_config()
        if self.config == 0:
            raise RuntimeError("Error creating VkFFTConfiguration")
        self.app = _vkfft_cuda.init_app(self.config)
        if self.app == 0:
            raise RuntimeError("Error creating VkFFTApplication. Was the CUDA driver initialised .")


    def __del__(self):
        """ Takes care of deleting allocated memory in the underlying
        VkFFTApplication and VkFFTConfiguration.
        """
        _vkfft_cuda.free_app(self.app)
        _vkfft_cuda.free_config(self.config)

    def _make_config(self):
        """ Create a vkfft configuration for a FFT transform"""
        nx, ny, nz = 1, 1, 1
        if self.d.ndim == 3:
            nz, ny, nx = self.d.shape
        elif self.d.ndim == 2:
            ny, nx = self.d.shape
        elif self.d.ndim == 1:
            nx = self.d.shape[0]
        config = _vkfft_cuda.make_config(nx, ny, nz, self.ndim, int(self.d.gpudata))
        return config

    def fft(self):
        """
        COmpute the forward FFT
        :return: nothing
        """
        _vkfft_cuda.fft(self.app)

    def ifft(self):
        """
        Compute the backward FFT
        :return: nothing
        """
        _vkfft_cuda.ifft(self.app)


def _test(size):
    """ This will launch a simple 1D FFT test
    """
    return _vkfft_cuda.test_vkfft_cuda(size)
