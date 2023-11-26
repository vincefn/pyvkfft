# -*- coding: utf-8 -*-

# PyVkFFT
#   (c) 2021- : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

__all__ = ['fftn', 'ifftn', 'rfftn', 'irfftn', 'dctn', 'idctn', 'dstn', 'idstn',
           'vkfft_version', 'clear_vkfftapp_cache',
           'has_pycuda', 'has_opencl', 'has_cupy']

from enum import Enum
from functools import lru_cache
import numpy as np
from .base import complex32
from . import config

try:
    from .cuda import VkFFTApp as VkFFTApp_cuda, has_pycuda, has_cupy, vkfft_version

    if has_pycuda:
        import pycuda.gpuarray as cua
        import pycuda.driver as cu_drv
    if has_cupy:
        import cupy as cp
except (ImportError, OSError):
    has_cupy, has_pycuda = False, False

try:
    from .opencl import VkFFTApp as VkFFTApp_cl, cla, vkfft_version

    has_opencl = True
except (ImportError, OSError):
    has_opencl = False


class Backend(Enum):
    """ Backend language & library"""
    UNKNOWN = 0
    PYCUDA = 1
    PYOPENCL = 2
    CUPY = 3


def _prepare_transform(src, dest, cl_queue, cuda_stream, r2c=False, r2c_odd=False):
    """
    Determine the backend from the input data.
    Create the destination array if necessary.

    :param src: the source GPU array
    :param dest: the destination array. If None, a new GPU array is created.
    :param cl_queue: the opencl queue to use, or None
    :param cuda_stream: the cuda stream to use (from cupy or pycuda), or None
    :param r2c: if True, this is for an R2C transform, so adapt the destination
        array accordingly.
    :param r2c_odd: True if the r2c transform has an odd size (for the real
        array) along the x-axis. This is only needed for a c2r out-of-place
        transform to determine the destination x-axis size.
    :return: a tuple (backend, inplace, dest, cl_queue, cuda_stream, devctx),
        also appending the destination dtype for a r2c transform.
        devctx is either the device or context unique ptr or id - this
        is only used to cache the VkFFTapp (e.g. making sure the app is
        re-instantiated if the cuda device changes). The cuda_stream
        output will be the int ptr.
    """
    backend = Backend.UNKNOWN
    fastidx = np.argmin(src.strides)  # fast axis is the last only for C-ordered arrays
    if fastidx == src.ndim - 1:
        order = 'C'
    else:
        order = 'F'
    if r2c:
        if src.dtype in [np.float16, np.float32, np.float64]:
            sh = list(src.shape)
            sh[fastidx] = sh[fastidx] // 2 + 1
            dtype = np.complex64
            if src.dtype == np.float16:
                dtype = complex32
            elif src.dtype == np.float64:
                dtype = np.complex128
        else:
            sh = list(src.shape)
            sh[fastidx] = (sh[fastidx] - 1) * 2
            if r2c_odd:
                # This is only true for an out-of-place transform,
                # but sh is only used if dest is None
                sh[fastidx] += 1
            dtype = np.float32
            if src.dtype == complex32:
                dtype = np.float16
            elif src.dtype == np.complex128:
                dtype = np.float64
    else:
        sh, dtype = None, None
    devctx = None
    if has_pycuda:
        if isinstance(src, cua.GPUArray):
            backend = Backend.PYCUDA
            # Must cast the gpudata to int as it can either be a DeviceAllocation object
            # or an int (e.g. when using a view of another array)
            src_ptr = int(src.gpudata)
            if dest is None:
                if r2c:
                    dest = cua.empty(tuple(sh), dtype=dtype, allocator=src.allocator, order=order)
                else:
                    dest = cua.empty_like(src)
            dest_ptr = int(dest.gpudata)
            if cuda_stream is None:
                devctx = cu_drv.Context.get_current()
            elif isinstance(cuda_stream, cu_drv.Stream):
                # Pass an int to make sure it is hashable
                cuda_stream = cuda_stream.handle

    if backend == Backend.UNKNOWN and has_opencl:
        if isinstance(src, cla.Array):
            backend = Backend.PYOPENCL
            src_ptr = src.data.int_ptr
            if dest is None:
                if r2c:
                    dest = cla.empty(src.queue, tuple(sh), dtype=dtype, allocator=src.allocator, order=order)
                else:
                    dest = cla.empty_like(src)
            dest_ptr = dest.data.int_ptr
            if cl_queue is None:
                cl_queue = src.queue
            devctx = cl_queue.context

    if backend == Backend.UNKNOWN and has_cupy:
        if isinstance(src, cp.ndarray):
            backend = Backend.CUPY
            src_ptr = src.__cuda_array_interface__['data'][0]
            if dest is None:
                if r2c:
                    dest = cp.empty(tuple(sh), dtype=dtype, order=order)
                else:
                    dest = cp.empty_like(src)
            dest_ptr = dest.__cuda_array_interface__['data'][0]
            if cuda_stream is None:
                cuda_stream = cp.cuda.get_current_stream()
            if isinstance(cuda_stream, cp.cuda.Stream):
                cuda_stream = cuda_stream.ptr
            devctx = cp.cuda.Device().id

    if backend == Backend.UNKNOWN:
        raise RuntimeError("Could note determine the type of GPU array supplied, or the "
                           "corresponding backend is not installed "
                           "(has_pycuda=%d, has_pyopencl=%d, has_cupy=%d)" %
                           (has_pycuda, has_opencl, has_cupy))

    inplace = dest_ptr == src_ptr
    if r2c:
        if inplace:
            dest = src.view(dtype=dtype)
        return backend, inplace, dest, cl_queue, cuda_stream, devctx, dtype
    else:
        return backend, inplace, dest, cl_queue, cuda_stream, devctx


@lru_cache(maxsize=config.FFT_CACHE_NB)
def _get_fft_app(backend, shape, dtype, inplace, ndim, axes, norm, cuda_stream, cl_queue,
                 devctx, strides=None, tune=False):
    del devctx  # Variable is just used for proper lru_cache
    sback = {Backend.PYCUDA: 'pycuda', Backend.CUPY: 'cupy', Backend.PYOPENCL: 'pyopencl'}[backend]
    tune_config = {'backend': sback} if tune else None
    if backend in [Backend.PYCUDA, Backend.CUPY]:
        return VkFFTApp_cuda(shape, dtype, ndim=ndim, inplace=inplace,
                             stream=cuda_stream, norm=norm, axes=axes, strides=strides,
                             tune_config=tune_config)
    elif backend == Backend.PYOPENCL:
        return VkFFTApp_cl(shape, dtype, cl_queue, ndim=ndim, inplace=inplace,
                           norm=norm, axes=axes, strides=strides,
                           tune_config=tune_config)


@lru_cache(maxsize=config.FFT_CACHE_NB)
def _get_rfft_app(backend, shape, dtype, inplace, ndim, norm, cuda_stream, cl_queue,
                  devctx, strides=None, tune=False, r2c_odd=False):
    del devctx  # Variable is just used for proper lru_cache
    sback = {Backend.PYCUDA: 'pycuda', Backend.CUPY: 'cupy', Backend.PYOPENCL: 'pyopencl'}[backend]
    tune_config = {'backend': sback} if tune else None
    if backend in [Backend.PYCUDA, Backend.CUPY]:
        return VkFFTApp_cuda(shape, dtype, ndim=ndim, inplace=inplace,
                             stream=cuda_stream, norm=norm, r2c=True, strides=strides,
                             tune_config=tune_config, r2c_odd=r2c_odd)
    elif backend == Backend.PYOPENCL:
        return VkFFTApp_cl(shape, dtype, cl_queue, ndim=ndim, inplace=inplace,
                           norm=norm, r2c=True, strides=strides,
                           tune_config=tune_config, r2c_odd=r2c_odd)


@lru_cache(maxsize=config.FFT_CACHE_NB)
def _get_dct_app(backend, shape, dtype, inplace, ndim, norm, dct_type,
                 cuda_stream, cl_queue, devctx, tune=False):
    del devctx  # Variable is just used for proper lru_cache
    sback = {Backend.PYCUDA: 'pycuda', Backend.CUPY: 'cupy', Backend.PYOPENCL: 'pyopencl'}[backend]
    tune_config = {'backend': sback} if tune else None
    if backend in [Backend.PYCUDA, Backend.CUPY]:
        return VkFFTApp_cuda(shape, dtype, ndim=ndim, inplace=inplace,
                             stream=cuda_stream, norm=norm, dct=dct_type,
                             tune_config=tune_config)
    elif backend == Backend.PYOPENCL:
        return VkFFTApp_cl(shape, dtype, cl_queue, ndim=ndim, inplace=inplace,
                           norm=norm, dct=dct_type,
                           tune_config=tune_config)


@lru_cache(maxsize=config.FFT_CACHE_NB)
def _get_dst_app(backend, shape, dtype, inplace, ndim, norm, dst_type,
                 cuda_stream, cl_queue, devctx, tune=False):
    del devctx  # Variable is just used for proper lru_cache
    sback = {Backend.PYCUDA: 'pycuda', Backend.CUPY: 'cupy', Backend.PYOPENCL: 'pyopencl'}[backend]
    tune_config = {'backend': sback} if tune else None
    if backend in [Backend.PYCUDA, Backend.CUPY]:
        return VkFFTApp_cuda(shape, dtype, ndim=ndim, inplace=inplace,
                             stream=cuda_stream, norm=norm, dst=dst_type,
                             tune_config=tune_config)
    elif backend == Backend.PYOPENCL:
        return VkFFTApp_cl(shape, dtype, cl_queue, ndim=ndim, inplace=inplace,
                           norm=norm, dst=dst_type,
                           tune_config=tune_config)


def fftn(src, dest=None, ndim=None, norm=1, axes=None, cuda_stream=None, cl_queue=None,
         return_scale=False, tune=False):
    """
    Perform a FFT on a GPU array, automatically creating the VkFFTApp
    and caching it for future re-use.

    :param src: the source pycuda.gpuarray.GPUArray or cupy.ndarray
    :param dest: the destination GPU array. If None, a new GPU array will
        be created and returned (using the source array allocator
        (pycuda, pyopencl) if available).
        If dest is the same array as src, an inplace transform is done.
    :param ndim: the number of dimensions (<=3) to use for the FFT. By default,
        uses the array dimensions. Can be smaller, e.g. ndim=2 for a 3D
        array to perform a batched 3D FFT on all the layers. The FFT
        is always performed along the last axes if the array's number
        of dimension is larger than ndim, i.e. on the x-axis for ndim=1,
        on the x and y axes for ndim=2.
    :param norm: if 0 (un-normalised), every transform multiplies the L2 norm
        of the array by the transform size.
        if 1 (the default) or "backward", the inverse transform divides the
        L2 norm by the array size, so FFT+iFFT will keep the array norm.
        if "ortho", each transform will keep the L2 norm, but that will
        involve an extra read & write operation.
    :param axes: a list or tuple of axes along which the transform is made.
        if None, the transform is done along the ndim fastest axes, or all
        axes if ndim is None. Not allowed for R2C transforms
    :param cuda_stream: the pycuda.driver.Stream or cupy.cuda.Stream to use
        for the transform. If None, the default one will be used
    :param cl_queue: the pyopencl.CommandQueue to be used. If None,
        the source array default queue will be used
    :param return_scale: if True, return the scale factor by which the result
        must be multiplied to keep its L2 norm after the transform
    :param tune: if True, will activate the automatic tuning of VkFFT
        parameters to maximise the FT throughput. This uses a quick
        approach testing a few transforms (about 4) before choosing the
        optimal parameters. This is similar to FFTW's FFTW_MEASURE approach.
    :return: the destination array if return_scale is False, or (dest, scale)
    """
    backend, inplace, dest, cl_queue, cuda_stream, devctx = _prepare_transform(src, dest, cl_queue, cuda_stream, False)
    app = _get_fft_app(backend, src.shape, src.dtype, inplace, ndim, axes, norm, cuda_stream, cl_queue, devctx,
                       strides=src.strides, tune=tune)
    if backend == Backend.PYOPENCL:
        app.fft(src, dest, queue=cl_queue)
    else:
        app.fft(src, dest)
    if return_scale:
        s = app.get_fft_scale()
        return dest, s
    return dest


def ifftn(src, dest=None, ndim=None, norm=1, axes=None, cuda_stream=None, cl_queue=None,
          return_scale=False, tune=False):
    """
    Perform an inverse FFT on a GPU array, automatically creating the VkFFTApp
    and caching it for future re-use.

    :param src: the source pycuda.gpuarray.GPUArray or cupy.ndarray
    :param dest: the destination GPU array. If None, a new GPU array will
        be created and returned (using the source array allocator
        (pycuda, pyopencl) if available).
        If dest is the same array as src, an inplace transform is done.
    :param ndim: the number of dimensions (<=3) to use for the FFT. By default,
        uses the array dimensions. Can be smaller, e.g. ndim=2 for a 3D
        array to perform a batched 3D FFT on all the layers. The FFT
        is always performed along the last axes if the array's number
        of dimension is larger than ndim, i.e. on the x-axis for ndim=1,
        on the x and y axes for ndim=2.
    :param norm: if 0 (un-normalised), every transform multiplies the L2 norm
        of the array by the transform size.
        if 1 (the default) or "backward", the inverse transform divides the
        L2 norm by the array size, so FFT+iFFT will keep the array norm.
        if "ortho", each transform will keep the L2 norm, but that will
        involve an extra read & write operation.
    :param axes: a list or tuple of axes along which the transform is made.
        if None, the transform is done along the ndim fastest axes, or all
        axes if ndim is None. Not allowed for R2C transforms
    :param cuda_stream: the pycuda.driver.Stream or cupy.cuda.Stream to use
        for the transform. If None, the default one will be used
    :param cl_queue: the pyopencl.CommandQueue to be used. If None,
        the source array default queue will be used
    :param return_scale: if True, return the scale factor by which the result
        must be multiplied to keep its L2 norm after the transform
    :param tune: if True, will activate the automatic tuning of VkFFT
        parameters to maximise the FT throughput. This uses a quick
        approach testing a few transforms (about 4) before choosing the
        optimal parameters. This is similar to FFTW's FFTW_MEASURE approach.
    :return: the destination array if return_scale is False, or (dest, scale)
    """
    backend, inplace, dest, cl_queue, cuda_stream, devctx = _prepare_transform(src, dest, cl_queue, cuda_stream, False)
    app = _get_fft_app(backend, src.shape, src.dtype, inplace, ndim, axes, norm, cuda_stream, cl_queue, devctx,
                       strides=src.strides, tune=tune)
    if backend == Backend.PYOPENCL:
        app.ifft(src, dest, queue=cl_queue)
    else:
        app.ifft(src, dest)
    if return_scale:
        s = app.get_fft_scale()
        return dest, s
    return dest


def rfftn(src, dest=None, ndim=None, norm=1, cuda_stream=None, cl_queue=None,
          return_scale=False, tune=False, r2c_odd=False):
    """
    Perform a real->complex transform on a GPU array, automatically creating
    the VkFFTApp and caching it for future re-use.
    For an out-of-place transform, the length of the destination last axis will
    be src.shape[-1]//2+1.
    For an in-place transform with an even [respectively odd]-sized
    fast (x) axis, the src array should have a shape (..., nx+2)
    [respectively (..., nx+1)], the last one or two values along the
    fast (x) axis are ignored, and the destination
    array will have a shape of (..., nx//2+1).
    An in-place transform with an odd-sized x-axis requires r2c_odd=True

    :param src: the source pycuda.gpuarray.GPUArray or cupy.ndarray
    :param dest: the destination GPU array. If None, a new GPU array will
        be created and returned (using the source array allocator
        (pycuda, pyopencl) if available).
        If dest is the same array as src, an inplace transform is done.
    :param ndim: the number of dimensions (<=3) to use for the FFT. By default,
        uses the array dimensions. Can be smaller, e.g. ndim=2 for a 3D
        array to perform a batched 3D FFT on all the layers. The FFT
        is always performed along the last axes if the array's number
        of dimension is larger than ndim, i.e. on the x-axis for ndim=1,
        on the x and y axes for ndim=2.
    :param norm: if 0 (un-normalised), every transform multiplies the L2 norm
        of the array by the transform size.
        if 1 (the default) or "backward", the inverse transform divides the
        L2 norm by the array size, so FFT+iFFT will keep the array norm.
        if "ortho", each transform will keep the L2 norm, but that will
        involve an extra read & write operation.
    :param cuda_stream: the pycuda.driver.Stream or cupy.cuda.Stream to use
        for the transform. If None, the default one will be used
    :param cl_queue: the pyopencl.CommandQueue to be used. If None,
        the source array default queue will be used
    :param return_scale: if True, return the scale factor by which the result
        must be multiplied to keep its L2 norm after the transform
    :param tune: if True, will activate the automatic tuning of VkFFT
        parameters to maximise the FT throughput. This uses a quick
        approach testing a few transforms (about 4) before choosing the
        optimal parameters. This is similar to FFTW's FFTW_MEASURE approach.
    :param r2c_odd: should be set to True for an in-place r2c transform
        where the actual data length is odd along the fast axis. This
        parameter is ignored otherwise.
    :return: the destination array if return_scale is False, or (dest, scale).
        For an in-place transform, the returned value is a view of the array
        with the appropriate type.
    """
    backend, inplace, dest, cl_queue, cuda_stream, devctx, dtype = \
        _prepare_transform(src, dest, cl_queue, cuda_stream, True)
    app = _get_rfft_app(backend, src.shape, src.dtype, inplace, ndim, norm, cuda_stream, cl_queue, devctx,
                        strides=src.strides, tune=tune, r2c_odd=r2c_odd)
    if backend == Backend.PYOPENCL:
        app.fft(src, dest, queue=cl_queue)
    else:
        app.fft(src, dest)
    if return_scale:
        s = app.get_fft_scale()
        return dest.view(dtype=dtype), s
    return dest.view(dtype=dtype)


def irfftn(src, dest=None, ndim=None, norm=1, cuda_stream=None, cl_queue=None,
           return_scale=False, tune=False, r2c_odd=False):
    """
    Perform a complex->real transform on a GPU array, automatically creating
    the VkFFTApp and caching it for future re-use.
    For an out-of-place transform, the length of the destination last axis will
    be (src.shape[-1]-1)*2.
    For an in-place transform, if the src complex array has a shape (..., nx),
    the destination (real) array will have a shape of (..., nx*2), but the last
    one (if r2c_odd=True) or two values along the x-axis are used as buffer:
    the size of the transform is thus either nx*2 or nx*2+1.

    :param src: the source pycuda.gpuarray.GPUArray or cupy.ndarray
    :param dest: the destination GPU array. If None, a new GPU array will
        be created and returned (using the source array allocator
        (pycuda, pyopencl) if available).
        If dest is the same array as src, an inplace transform is done.
    :param ndim: the number of dimensions (<=3) to use for the FFT. By default,
        uses the array dimensions. Can be smaller, e.g. ndim=2 for a 3D
        array to perform a batched 3D FFT on all the layers. The FFT
        is always performed along the last axes if the array's number
        of dimension is larger than ndim, i.e. on the x-axis for ndim=1,
        on the x and y axes for ndim=2.
    :param norm: if 0 (un-normalised), every transform multiplies the L2 norm
        of the array by the transform size.
        if 1 (the default) or "backward", the inverse transform divides the
        L2 norm by the array size, so FFT+iFFT will keep the array norm.
        if "ortho", each transform will keep the L2 norm, but that will
        involve an extra read & write operation.
    :param cuda_stream: the pycuda.driver.Stream or cupy.cuda.Stream to use
        for the transform. If None, the default one will be used
    :param cl_queue: the pyopencl.CommandQueue to be used. If None,
        the source array default queue will be used
    :param return_scale: if True, return the scale factor by which the result
        must be multiplied to keep its L2 norm after the transform
    :param tune: if True, will activate the automatic tuning of VkFFT
        parameters to maximise the FT throughput. This uses a quick
        approach testing a few transforms (about 4) before choosing the
        optimal parameters. This is similar to FFTW's FFTW_MEASURE approach.
    :param r2c_odd: should be set to True for an in-place r2c transform
        where the actual data length (in the real array) is odd along the
        fast axis. This parameter is ignored otherwise.
    :return: the destination array if return_scale is False, or (dest, scale)
        For an in-place transform, the returned value is a view of the array
        with the appropriate type.
    """
    backend, inplace, dest, cl_queue, cuda_stream, devctx, dtype = \
        _prepare_transform(src, dest, cl_queue, cuda_stream, True, r2c_odd=r2c_odd)
    app = _get_rfft_app(backend, dest.shape, dest.dtype, inplace, ndim, norm, cuda_stream, cl_queue, devctx,
                        strides=dest.strides, tune=tune, r2c_odd=r2c_odd)
    if backend == Backend.PYOPENCL:
        app.ifft(src, dest, queue=cl_queue)
    else:
        app.ifft(src, dest)
    if return_scale:
        s = app.get_fft_scale()
        return dest.view(dtype=dtype), s
    return dest.view(dtype=dtype)


def dctn(src, dest=None, ndim=None, norm=1, dct_type=2, cuda_stream=None, cl_queue=None, tune=False):
    """
    Perform a real->real Direct Cosine Transform on a GPU array, automatically
    creating the VkFFTApp and caching it for future re-use.

    :param src: the source pycuda.gpuarray.GPUArray or cupy.ndarray
    :param dest: the destination GPU array. If None, a new GPU array will
        be created and returned (using the source array allocator
        (pycuda, pyopencl) if available).
        If dest is the same array as src, an inplace transform is done.
    :param ndim: the number of dimensions (<=3) to use for the FFT. By default,
        uses the array dimensions. Can be smaller, e.g. ndim=2 for a 3D
        array to perform a batched 3D FFT on all the layers. The FFT
        is always performed along the last axes if the array's number
        of dimension is larger than ndim, i.e. on the x-axis for ndim=1,
        on the x and y axes for ndim=2.
    :param norm: normalisation mode, either 0 (un-normalised) or
        1 (the default, also available as "backward) which will normalise
        the inverse transform, so DCT+iDCT will keep the array norm.
    :param dct_type: the type of dct desired: 1, 2 (default), 3 or 4
    :param cuda_stream: the pycuda.driver.Stream or cupy.cuda.Stream to use
        for the transform. If None, the default one will be used
    :param cl_queue: the pyopencl.CommandQueue to be used. If None,
        the source array default queue will be used
    :param tune: if True, will activate the automatic tuning of VkFFT
        parameters to maximise the FT throughput. This uses a quick
        approach testing a few transforms (about 4) before choosing the
        optimal parameters. This is similar to FFTW's FFTW_MEASURE approach.
    :return: the destination array.
    """
    backend, inplace, dest, cl_queue, cuda_stream, devctx = _prepare_transform(src, dest, cl_queue, cuda_stream, False)
    app = _get_dct_app(backend, src.shape, src.dtype, inplace, ndim, norm,
                       dct_type, cuda_stream, cl_queue, devctx, tune=tune)
    if backend == Backend.PYOPENCL:
        app.fft(src, dest, queue=cl_queue)
    else:
        app.fft(src, dest)
    return dest


def idctn(src, dest=None, ndim=None, norm=1, dct_type=2, cuda_stream=None, cl_queue=None, tune=False):
    """
    Perform a real->real inverse Direct Cosine Transform on a GPU array,
    automatically creating the VkFFTApp and caching it for future re-use.

    :param src: the source pycuda.gpuarray.GPUArray or cupy.ndarray
    :param dest: the destination GPU array. If None, a new GPU array will
        be created and returned (using the source array allocator
        (pycuda, pyopencl) if available).
        If dest is the same array as src, an inplace transform is done.
    :param ndim: the number of dimensions (<=3) to use for the FFT. By default,
        uses the array dimensions. Can be smaller, e.g. ndim=2 for a 3D
        array to perform a batched 3D FFT on all the layers. The FFT
        is always performed along the last axes if the array's number
        of dimension is larger than ndim, i.e. on the x-axis for ndim=1,
        on the x and y axes for ndim=2.
    :param norm: normalisation mode, either 0 (un-normalised) or
        1 (the default, also available as "backward) which will normalise
        the inverse transform, so DCT+iDCT will keep the array norm.
    :param dct_type: the type of dct desired: 2 (default), 3 or 4
    :param cuda_stream: the pycuda.driver.Stream or cupy.cuda.Stream to use
        for the transform. If None, the default one will be used
    :param cl_queue: the pyopencl.CommandQueue to be used. If None,
        the source array default queue will be used
    :param tune: if True, will activate the automatic tuning of VkFFT
        parameters to maximise the FT throughput. This uses a quick
        approach testing a few transforms (about 4) before choosing the
        optimal parameters. This is similar to FFTW's FFTW_MEASURE approach.
    :return: the destination array.
    """
    backend, inplace, dest, cl_queue, cuda_stream, devctx = _prepare_transform(src, dest, cl_queue, cuda_stream, False)
    app = _get_dct_app(backend, src.shape, src.dtype, inplace, ndim, norm,
                       dct_type, cuda_stream, cl_queue, devctx, tune=tune)
    if backend == Backend.PYOPENCL:
        app.ifft(src, dest, queue=cl_queue)
    else:
        app.ifft(src, dest)
    return dest


def dstn(src, dest=None, ndim=None, norm=1, dst_type=2, cuda_stream=None, cl_queue=None, tune=False):
    """
    Perform a real->real Direct Cosine Transform on a GPU array, automatically
    creating the VkFFTApp and caching it for future re-use.

    :param src: the source pycuda.gpuarray.GPUArray or cupy.ndarray
    :param dest: the destination GPU array. If None, a new GPU array will
        be created and returned (using the source array allocator
        (pycuda, pyopencl) if available).
        If dest is the same array as src, an inplace transform is done.
    :param ndim: the number of dimensions (<=3) to use for the FFT. By default,
        uses the array dimensions. Can be smaller, e.g. ndim=2 for a 3D
        array to perform a batched 3D FFT on all the layers. The FFT
        is always performed along the last axes if the array's number
        of dimension is larger than ndim, i.e. on the x-axis for ndim=1,
        on the x and y axes for ndim=2.
    :param norm: normalisation mode, either 0 (un-normalised) or
        1 (the default, also available as "backward) which will normalise
        the inverse transform, so DST+iDST will keep the array norm.
    :param dst_type: the type of dst desired: 1, 2 (default), 3 or 4
    :param cuda_stream: the pycuda.driver.Stream or cupy.cuda.Stream to use
        for the transform. If None, the default one will be used
    :param cl_queue: the pyopencl.CommandQueue to be used. If None,
        the source array default queue will be used
    :param tune: if True, will activate the automatic tuning of VkFFT
        parameters to maximise the FT throughput. This uses a quick
        approach testing a few transforms (about 4) before choosing the
        optimal parameters. This is similar to FFTW's FFTW_MEASURE approach.
    :return: the destination array.
    """
    backend, inplace, dest, cl_queue, cuda_stream, devctx = _prepare_transform(src, dest, cl_queue, cuda_stream, False)
    app = _get_dst_app(backend, src.shape, src.dtype, inplace, ndim, norm,
                       dst_type, cuda_stream, cl_queue, devctx, tune=tune)
    if backend == Backend.PYOPENCL:
        app.fft(src, dest, queue=cl_queue)
    else:
        app.fft(src, dest)
    return dest


def idstn(src, dest=None, ndim=None, norm=1, dst_type=2, cuda_stream=None, cl_queue=None, tune=False):
    """
    Perform a real->real inverse Direct Cosine Transform on a GPU array,
    automatically creating the VkFFTApp and caching it for future re-use.

    :param src: the source pycuda.gpuarray.GPUArray or cupy.ndarray
    :param dest: the destination GPU array. If None, a new GPU array will
        be created and returned (using the source array allocator
        (pycuda, pyopencl) if available).
        If dest is the same array as src, an inplace transform is done.
    :param ndim: the number of dimensions (<=3) to use for the FFT. By default,
        uses the array dimensions. Can be smaller, e.g. ndim=2 for a 3D
        array to perform a batched 3D FFT on all the layers. The FFT
        is always performed along the last axes if the array's number
        of dimension is larger than ndim, i.e. on the x-axis for ndim=1,
        on the x and y axes for ndim=2.
    :param norm: normalisation mode, either 0 (un-normalised) or
        1 (the default, also available as "backward) which will normalise
        the inverse transform, so DST+iDST will keep the array norm.
    :param dst_type: the type of dst desired: 2 (default), 3 or 4
    :param cuda_stream: the pycuda.driver.Stream or cupy.cuda.Stream to use
        for the transform. If None, the default one will be used
    :param cl_queue: the pyopencl.CommandQueue to be used. If None,
        the source array default queue will be used
    :param tune: if True, will activate the automatic tuning of VkFFT
        parameters to maximise the FT throughput. This uses a quick
        approach testing a few transforms (about 4) before choosing the
        optimal parameters. This is similar to FFTW's FFTW_MEASURE approach.
    :return: the destination array.
    """
    backend, inplace, dest, cl_queue, cuda_stream, devctx = _prepare_transform(src, dest, cl_queue, cuda_stream, False)
    app = _get_dst_app(backend, src.shape, src.dtype, inplace, ndim, norm,
                       dst_type, cuda_stream, cl_queue, devctx, tune=tune)
    if backend == Backend.PYOPENCL:
        app.ifft(src, dest, queue=cl_queue)
    else:
        app.ifft(src, dest)
    return dest


def clear_vkfftapp_cache():
    """ Remove all cached VkFFTApp"""
    _get_fft_app.cache_clear()
    _get_rfft_app.cache_clear()
