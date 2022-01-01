# -*- coding: utf-8 -*-

# PyVkFFT
#   (c) 2021- : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr
#
#
# Functions for accuracy tests.

import numpy as np
import timeit
from numpy.fft import fftn, ifftn, rfftn, irfftn

try:
    # We prefer scipy over numpy for fft, and we can also test dct
    from scipy.fft import dctn, idctn, fftn, ifftn, rfftn, irfftn

    has_scipy_dct = True
except ImportError:
    has_scipy_dct = False
    print("Install scipy if you want to test dct transforms")

try:
    from pyfftw.interfaces.scipy_fft import dctn, idctn, fftn, ifftn, rfftn, irfftn

    has_pyfftw = True
except ImportError:
    has_pyfftw = False
    print("Install pyfftw if you want greater accuracy tests")

try:
    import pyopencl as cl
    import pyopencl.array as cla
    from pyvkfft.opencl import VkFFTApp as clVkFFTApp, primes

    has_opencl = True
except ImportError:
    has_opencl = False

try:
    from pyvkfft.cuda import VkFFTApp as cuVkFFTApp, primes, has_pycuda, has_cupy

    if has_pycuda:
        import pycuda.autoinit
        import pycuda.driver as cu_drv
        import pycuda.gpuarray as cua

    if has_cupy:
        import cupy as cp
except ImportError:
    has_cupy = False
    has_pycuda = False


def l2(a, b):
    """L2 norm"""
    return np.sqrt((abs(a - b) ** 2).sum() / (abs(a) ** 2).sum())


def li(a, b):
    """Linf norm"""
    return abs(a - b).max() / abs(a).max()


def test_accuracy(backend, shape, ndim, axes, dtype, inplace, norm, use_lut, r2c=False, dct=False,
                  stream=None, queue=None, return_array=False, init_array=None, verbose=False):
    """
    Measure the
    :param backend: either 'pyopencl', 'pycuda' or 'cupy'
    :param shape: the shape of the array to test. If this is an inplace r2c, the
        x-axis length must be even, and two extra values will be appended along x,
        so the actual transform shape is the one supplied
    :param ndim: the number of FFT dimensions. Can be None if axes is given
    :param axes: the transform axes. Supersedes ndim
    :param dtype: either np.complex64 or np.complex128, or np.float32/np.float64 for r2c & dct
    :param inplace: if True, make an inplace transform. Note that for inplace r2c transforms,
        the size for the last (x, fastest) axis must be even.
    :param norm: either 0, 1 or "ortho"
    :param use_lut: if True,1, False or 0, will trigger useLUT=1 or 0 for VkFFT.
        If None, the default VkFFT behaviour is used.
    :param r2c: if True, test an r2c transform. If inplace, the last dimension
        (x, fastest axis) must be even
    :param dct: either 1, 2, 3 or 4 to test different dct. Only norm=1 is can be
        tested (native scipy/pyfftw normalisation).
    :param stream: the cuda stream to use, or None
    :param queue: the opencl queue to use (mandatory for the 'pyopencl' backend)
    :param return_array: if True, will return the generated random array so it can be
        re-used for different parameters
    :param init_array: the initial  (numpy) random array to use (should be filled
        with uniform random numbers between +/-0.5 for both real and imaginary
        fields), to save time. The correct type will be applied.
        If None, a random array is generated.
    :param verbose: if True, print a 1-line info for both fft and ifft results
    :return: a tuple (l2_fft, li_fft, l2_ifft, li_ifft, tol, dt_array, dt_app, dt_fft,
        dt_ifft, src_unchanged_fft, src_unchanged_ifft, ok), with the L2 and Linf
        normalised norms comparing pyvkfft's result with either numpy, scipy or
        pyfftw (in long double precision for the latter), the reference tolerance, and
        the times spent in preparing the initial random array, creating the VkFFT app,
        and performing the forward and backward transforms (including the GPU and
        reference transforms, plus the L2 and Linf computations - don't use this for
        benchmarking), 'src_fft_unchanged' and 'srf_ifft_unchanged' are True if for
        an out-of-place transform, the source array is actually unmodified (which is
        not true for r2c ifft with ndim>=2).
        The final value 'ok' is True if both li_fft and li_ifft are smaller than tol.
        If return_array is True, the initial random array used is returned as the last
        element of the tuple
    """
    t0 = timeit.default_timer()
    if dtype in (np.complex64, np.float32):
        dtype = np.complex64
        dtypef = np.float32
    else:
        dtype = np.complex128
        dtypef = np.float64

    if dct:
        if norm != 1:
            raise RuntimeError("test_accuracy: only norm=1 can be used with dct")

    if r2c:
        if inplace:
            # Add two extra columns in the source array
            # so the transform has the desired shape
            shape = list(shape)
            shape[-1] += 2
        else:
            shapec = list(shape)
            shapec[-1] = shapec[-1] // 2 + 1
            shapec = tuple(shapec)
    else:
        shapec = tuple(shape)
    shape = tuple(shape)

    if init_array is not None:
        if r2c:
            if inplace:
                d0 = np.empty(shape, dtype=dtypef)
                d0[..., :-2] = init_array
            else:
                d0 = init_array.astype(dtypef)
        elif dct:
            d0 = init_array.astype(dtypef)
        else:
            d0 = init_array.astype(dtype)
    else:
        if r2c or dct:
            d0 = np.random.uniform(-0.5, 0.5, shape).astype(dtypef)
        else:
            d0 = (np.random.uniform(-0.5, 0.5, shape) + 1j * np.random.uniform(-0.5, 0.5, shape)).astype(dtype)

    t1 = timeit.default_timer()

    if 'opencl' in backend:
        app = clVkFFTApp(d0.shape, d0.dtype, queue, ndim=ndim, norm=norm,
                         axes=axes, useLUT=use_lut, inplace=inplace, r2c=r2c, dct=dct)
        t2 = timeit.default_timer()
        d_gpu = cla.to_device(queue, d0)
    else:
        if backend == "pycuda":
            to_gpu = cua.to_gpu
        else:
            to_gpu = cp.array

        app = cuVkFFTApp(d0.shape, d0.dtype, ndim=ndim, norm=norm, axes=axes,
                         useLUT=use_lut, inplace=inplace, r2c=r2c, dct=dct, stream=stream)
        t2 = timeit.default_timer()
        d_gpu = to_gpu(d0)

    if axes is None:
        axes_numpy = list(range(len(shape)))[-ndim:]
    else:
        axes_numpy = axes
    # base FFT scale for numpy (not used for DCT)
    s = np.sqrt(np.prod([d0.shape[i] for i in axes_numpy]))
    if r2c and inplace:
        s = np.sqrt(s ** 2 / d0.shape[-1] * (d0.shape[-1] - 2))

    # Tolerance estimated from accuracy notebook
    if dtype in (np.complex64, np.float32):
        tol = 1e-6 + 4e-7 * np.log10(s ** 2)
    else:
        tol = 5e-15 + 5e-16 * np.log10(s ** 2)

    # FFT
    if inplace:
        d1_gpu = d_gpu
    else:
        if r2c:
            if backend == "pyopencl":
                d1_gpu = cla.empty(queue, shapec, dtype=dtype)
            elif backend == "pycuda":
                d1_gpu = cua.empty(shapec, dtype=dtype)
            elif backend == "cupy":
                d1_gpu = cp.empty(shapec, dtype=dtype)
        else:
            d1_gpu = d_gpu.copy()

    if has_pyfftw:
        if r2c or dct:
            d0n = d0.astype(np.longdouble)
        else:
            d0n = d0.astype(np.clongdouble)
    else:
        d0n = d0

    if r2c:
        if inplace:
            d = rfftn(d0n[..., :-2], axes=axes_numpy) / s
        else:
            d = rfftn(d0n, axes=axes_numpy) / s
    elif dct:
        d = dctn(d0n, axes=axes_numpy, type=dct)
    else:
        d = fftn(d0n, axes=axes_numpy) / s

    if dct:
        d1_gpu = app.fft(d_gpu, d1_gpu)
    else:
        d1_gpu = app.fft(d_gpu, d1_gpu) * app.get_fft_scale()

    if inplace and r2c:
        assert d1_gpu.dtype == dtype, "The array type is incorrect after an inplace FFT"

    n2, ni = l2(d, d1_gpu.get()), li(d, d1_gpu.get())

    src_unchanged_fft = np.all(np.equal(d_gpu.get(), d0))

    if verbose:
        if r2c:
            t = "R2C"
        elif dct:
            t = "DCT%d" % dct
        else:
            t = "C2C"
        if r2c and inplace:
            tmp = list(d0.shape)
            tmp[-1] -= 2
            shstr = str(tuple(tmp)).replace(" ", "")
            if "')" in shstr:
                shstr = shstr.replace(",)", "+2)")
            else:
                shstr = shstr.replace(")", "+2)")
        else:
            shstr = str(d0.shape).replace(" ", "")
        shax = str(axes).replace(" ", "")
        red = max(0, min(int((ni / tol - 0.2) * 255), 255))
        stol = "\x1b[38;2;%d;0;0m%6.2e < %6.2e (%5.3f)\x1b[0m" % (red, ni, tol, ni / tol)

        verb_out = "%8s %4s %13s axes=%10s ndim=%4s %10s lut=%4s inplace=%d " \
                   " norm=%4s %5s: n2=%6.2e ninf=%s %5s %d" % \
                   (backend, t, shstr, shax, str(ndim), str(d0.dtype),
                    str(use_lut), int(inplace), str(norm), "FFT", n2, stol, ni < tol, src_unchanged_fft)
    t3 = timeit.default_timer()

    # IFFT - from original array to avoid error propagation
    if r2c:
        # Exception: we need a proper half-Hermitian array
        d0 = d.astype(dtype)
        if has_pyfftw:
            d0n = d0.astype(np.clongdouble)
        else:
            d0n = d0

    if 'opencl' in backend:
        d_gpu = cla.to_device(queue, d0)
    else:
        d_gpu = to_gpu(d0)

    if inplace:
        d1_gpu = d_gpu
    else:
        if r2c:
            if backend == "pyopencl":
                d1_gpu = cla.empty(queue, shape, dtype=dtypef)
            elif backend == "pycuda":
                d1_gpu = cua.empty(shape, dtype=dtypef)
            elif backend == "cupy":
                d1_gpu = cp.empty(shape, dtype=dtypef)
        else:
            d1_gpu = d_gpu.copy()

    if r2c:
        d = irfftn(d0n, axes=axes_numpy) * s
    elif dct:
        d = idctn(d0n, axes=axes_numpy, type=dct)
    else:
        d = ifftn(d0n, axes=axes_numpy) * s

    if dct:
        d1_gpu = app.ifft(d_gpu, d1_gpu)
    else:
        d1_gpu = app.ifft(d_gpu, d1_gpu) * app.get_ifft_scale()

    if inplace:
        if dct or r2c:
            assert d1_gpu.dtype == dtypef, "The array type is incorrect after an inplace iFFT"
        else:
            assert d1_gpu.dtype == dtype, "The array type is incorrect after an inplace iFFT"

    if r2c and inplace:
        n2i, nii = l2(d, d1_gpu.get()[..., :-2]), li(d, d1_gpu.get()[..., :-2])
    else:
        n2i, nii = l2(d, d1_gpu.get()), li(d, d1_gpu.get())

    src_unchanged_ifft = np.all(np.equal(d_gpu.get(), d0))

    if verbose:
        red = max(0, min(int((nii / tol - 0.) * 2550), 255))
        stol = "\x1b[38;2;%d;0;0m%6.2e < %6.2e (%5.3f)\x1b[0m" % (red, nii, tol, nii / tol)
        verb_out += "%5s: n2=%6.2e ninf=%s %5s %d" % ("iFFT", n2i, stol, nii < tol, src_unchanged_ifft)
        print(verb_out)
    t4 = timeit.default_timer()

    if return_array:
        return n2, ni, n2i, nii, tol, t1 - t0, t2 - t1, t3 - t2, t4 - t3, \
               src_unchanged_fft, src_unchanged_ifft, max(ni, nii) < tol, d0
    return n2, ni, n2i, nii, tol, t1 - t0, t2 - t1, t3 - t2, t4 - t3, \
           src_unchanged_fft, src_unchanged_ifft, max(ni, nii) < tol
