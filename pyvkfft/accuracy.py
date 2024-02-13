# -*- coding: utf-8 -*-

# PyVkFFT
#   (c) 2021- : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr
#
#
# Functions for accuracy tests.

import os
import platform
import multiprocessing
import timeit
import atexit

import psutil
import numpy as np
from numpy.fft import fftn, ifftn, rfftn, irfftn
from pyvkfft.base import primes

try:
    # We prefer scipy over numpy for fft, and we can also test dct
    from scipy.fft import dctn, idctn, fftn, ifftn, rfftn, irfftn, dstn, idstn

    has_dct_ref = True
    has_scipy = True
except ImportError:
    has_dct_ref = False
    has_scipy = False

# pyfftw speed is not good compared to scipy, when using every transform once
# try:
#     from pyfftw.interfaces import scipy_fft as pyfftw_fft
#     if not has_scipy:
#         from pyfftw.interfaces.scipy_fft import dctn, idctn, fftn, ifftn, rfftn, irfftn
#
#     has_pyfftw = True
#     has_dct_ref = True
# except ImportError:
#     has_pyfftw = False

try:
    import pyopencl as cl
    import pyopencl.array as cla
    from pyvkfft.opencl import VkFFTApp as clVkFFTApp

    has_opencl = True
except ImportError:
    has_opencl = False

try:
    from pyvkfft.cuda import VkFFTApp as cuVkFFTApp, has_pycuda, has_cupy

    if has_pycuda:
        import pycuda.driver as cu_drv
        import pycuda.gpuarray as cua

    if has_cupy:
        import cupy as cp
except ImportError:
    has_cupy = False
    has_pycuda = False

# Dictionary of cuda/opencl (device, context). Will be initialised on-demand.
# This is needed for multiprocessing.
# The pyopencl entry is a tuple with (device, context, queue, has_cl_fp64)
gpu_ctx_dic = {}


def init_ctx(backend, gpu_name=None, opencl_platform=None, verbose=False):
    if backend in gpu_ctx_dic:
        return
    if backend == "pycuda":
        if not has_pycuda:
            raise RuntimeError("init_ctx: backend=%s is not available" % backend)
        cu_drv.init()
        d = None
        if gpu_name is not None:
            for i in range(cu_drv.Device.count()):
                if gpu_name.lower() in cu_drv.Device(i).name().lower():
                    d = cu_drv.Device(i)
                    break
        else:
            d = cu_drv.Device(0)
        if d is None:
            if gpu_name is not None:
                raise RuntimeError("Selected backend is pycuda, but no device found (name=%s)" % gpu_name)
            else:
                raise RuntimeError("Selected backend is pycuda, but no device found")
        gpu_ctx_dic["pycuda"] = (d, d.retain_primary_context())
        gpu_ctx_dic["pycuda"][1].push()
        if verbose:
            print("Selected device for pycuda: %s" % d.name())
    elif backend == "pyopencl":
        if not has_opencl:
            raise RuntimeError("init_ctx: backend=%s is not available" % backend)
        d = None
        for p in cl.get_platforms():
            if d is not None:
                break
            if opencl_platform is not None:
                if opencl_platform.lower() not in p.name.lower():
                    continue
            for d0 in p.get_devices():
                if d0.type & cl.device_type.GPU:
                    if gpu_name is not None:
                        if gpu_name.lower() in d0.name.lower():
                            d = d0
                    else:
                        d = d0
                if d is not None:
                    break
        if d is None:
            raise RuntimeError("Selected backend is pyopencl, but no device found (name=%s, platform=%s)" %
                               (gpu_name, opencl_platform))
        cl_ctx = cl.Context([d])
        cq = cl.CommandQueue(cl_ctx)
        gpu_ctx_dic["pyopencl"] = d, cl_ctx, cq, 'cl_khr_fp64' in cq.device.extensions
        if verbose:
            print("Selected device for pyopencl: %s [%s]" % (d.name, p.name))
    elif backend == "cupy":
        if not has_cupy:
            raise RuntimeError("init_ctx: backend=%s is not available" % backend)
        if gpu_name is not None:
            for i in range(cp.cuda.runtime.getDeviceCount()):
                if gpu_name.lower() in cp.cuda.runtime.getDeviceProperties(i)['name'].decode().lower():
                    d = cp.cuda.Device(i).use()
                    break
        else:
            d = cp.cuda.Device(0).use()
        if d is None:
            if gpu_name is not None:
                raise RuntimeError("Selected backend is pycuda, but no device found (name=%s)" % gpu_name)
            else:
                raise RuntimeError("Selected backend is pycuda, but no device found")
        gpu_ctx_dic["cupy"] = d

        # TODO: The following somehow helps initialising cupy, not sure why it's useful.
        #  (some context auto-init...). Otherwise a cuLaunchKernel error occurs with
        #  the first transform.
        cupy_a = cp.array(np.zeros((128, 128), dtype=np.float32))
        cupy_a.sum()
    else:
        raise RuntimeError("init_ctx: unknown backend ", backend)


def cleanup_cu_ctx():
    # Is that really clean ?
    if has_pycuda:
        if cu_drv.Context is not None:
            while cu_drv.Context.get_current() is not None:
                cu_drv.Context.pop()


atexit.register(cleanup_cu_ctx)


def l2(a, b):
    """L2 norm"""
    return np.sqrt((abs(a - b) ** 2).sum() / (abs(a) ** 2).sum())


def li(a, b):
    """Linf norm"""
    return abs(a - b).max() / abs(a).max()


def test_accuracy(backend, shape, ndim, axes, dtype, inplace, norm, use_lut,
                  r2c=False, dct=False, dst=False,
                  gpu_name=None, opencl_platform=None, stream=None, queue=None, return_array=False,
                  init_array=None, verbose=False, colour_output=False, ref_long_double=True, order='C'):
    """
    Measure the FT accuracy by comparing to the result from scipy (if available), or numpy.

    :param backend: either 'pyopencl', 'pycuda' or 'cupy'
    :param shape: the shape of the array to test. If this is an inplace r2c, the
        fast-axis length must be even, and two extra values will be appended along x,
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
        tested (native scipy normalisation).
    :param dst: either 1, 2, 3 or 4 to test different dst. Only norm=1 is can be
        tested (native scipy normalisation).
    :param gpu_name: the name of the gpu to use. If None, the first available
        for the backend will be used.
    :param opencl_platform: the name of the OpenCL platform to use. If None, the first available
        will be used.
    :param stream: the cuda stream to use, or None
    :param queue: the opencl queue to use (mandatory for the 'pyopencl' backend)
    :param return_array: if True, will return the generated random array so it can be
        re-used for different parameters
    :param init_array: the initial  (numpy) random array to use (should be filled
        with uniform random numbers between +/-0.5 for both real and imaginary
        fields), to save time. The correct type will be applied.
        If None, a random array is generated.
    :param verbose: if True, print a 1-line info for both fft and ifft results
    :param colour_output: if True, use some colour to tag the quality of the accuracy
    :param ref_long_double: if True and scipy is available, long double precision
        will be used for the reference transform. Otherwise, this is ignored.
    :param order: either 'C' (default C-contiguous) or 'F' to test a different
        stride. Note that for the latter, a 3D transform on a 4D array will not
        be supported as the last transform axis would be on the 4th dimension
        (once ordered by stride).
    :return: a dictionary with (l2_fft, li_fft, l2_ifft, li_ifft, tol, dt_array,
        dt_app, dt_fft, dt_ifft, src_unchanged_fft, src_unchanged_ifft, tol_test, str),
        with the L2 and Linf normalised norms comparing pyvkfft's result with either
        numpy, scipy, the reference
        tolerance, and the times spent in preparing the initial random array, creating
        the VkFFT app, and performing the forward and backward transforms (including
        the GPU and reference transforms, plus the L2 and Linf computations - don't use
        this for benchmarking), 'src_fft_unchanged' and 'srf_ifft_unchanged' are True
        if for an out-of-place transform, the source array is actually unmodified
        (which is not true for r2c ifft with ndim>=2).
        The last fields are 'tol_test' which is True if both li_fft and li_ifft are
        smaller than tol, and str the string summarising the results (printed if
        verbose is True). If return_array is True, the initial random array used is
        returned as 'd0'.
        All input parameters are also returned as key/values, except stream, queue,
        return_array, ini_array and verbose.
    """
    ndims = len(shape)
    if backend == "cupy" and has_cupy:
        mempool = cp.get_default_memory_pool()
        if mempool is not None:  # Is that test necessary ?
            # Clean memory pool, we are changing array sizes constantly, and using
            # N parallel process so memory management  must be done manually
            mempool.free_all_blocks()
    t0 = timeit.default_timer()
    init_ctx(backend, gpu_name=gpu_name, opencl_platform=opencl_platform, verbose=False)
    if backend == "pyopencl" and queue is None:
        queue = gpu_ctx_dic["pyopencl"][2]
    shape0 = shape
    dtype0 = dtype
    if dtype in (np.complex64, np.float32):
        dtype = np.complex64
        dtypef = np.float32
    else:
        dtype = np.complex128
        dtypef = np.float64

    if dct or dst:
        if norm != 1:
            raise RuntimeError("test_accuracy: only norm=1 can be used with dct or dst")
    if r2c:
        r2c_odd = shape[-1] % 2 == 1 if order == 'C' else shape[0] % 2 == 1
        r2c_inplace_pad = 1 if r2c_odd else 2
        if inplace:
            # Add two (or one for an odd-sized fast axis) extra columns in the source array
            # so the transform has the desired shape
            shape = list(shape)
            if order == 'C':
                shape[-1] += r2c_inplace_pad
            else:
                shape[0] += r2c_inplace_pad
        else:
            shapec = list(shape)
            if order == 'C':
                shapec[-1] = shapec[-1] // 2 + 1
            else:
                shapec[0] = shapec[0] // 2 + 1
            shapec = tuple(shapec)
    else:
        r2c_odd = None
        r2c_inplace_pad = None
        shapec = tuple(shape)
    shape = tuple(shape)

    if init_array is not None:
        if r2c:
            if inplace:
                d0 = np.empty(shape, dtype=dtypef)
                if order == 'C':
                    d0[..., :-r2c_inplace_pad] = init_array
                else:
                    d0[:-r2c_inplace_pad] = init_array
            else:
                d0 = init_array.astype(dtypef)
        elif dct or dst:
            d0 = init_array.astype(dtypef)
        else:
            d0 = init_array.astype(dtype)
    else:
        if r2c or dct or dst:
            d0 = np.random.uniform(-0.5, 0.5, shape).astype(dtypef)
        else:
            d0 = (np.random.uniform(-0.5, 0.5, shape) + 1j * np.random.uniform(-0.5, 0.5, shape)).astype(dtype)

    if order != 'C':
        d0 = np.asarray(d0, order=order)

    t1 = timeit.default_timer()

    if 'opencl' in backend:
        app = clVkFFTApp(d0.shape, d0.dtype, queue, ndim=ndim, norm=norm,
                         axes=axes, useLUT=use_lut, inplace=inplace,
                         r2c=r2c, dct=dct, dst=dst, strides=d0.strides,
                         r2c_odd=r2c_odd)
        t2 = timeit.default_timer()
        d_gpu = cla.to_device(queue, d0)
        empty_like = cla.empty_like
    else:
        if backend == "pycuda":
            to_gpu = cua.to_gpu
            empty_like = cua.empty_like
        else:
            to_gpu = cp.array
            empty_like = cp.empty_like

        app = cuVkFFTApp(d0.shape, d0.dtype, ndim=ndim, norm=norm, axes=axes,
                         useLUT=use_lut, inplace=inplace, r2c=r2c,
                         dct=dct, dst=dst, strides=d0.strides,
                         r2c_odd=r2c_odd, stream=stream)
        t2 = timeit.default_timer()
        d_gpu = to_gpu(d0)

    if axes is None:
        axes_numpy = list(range(ndims))[-ndim:]
    else:
        # Make sure axes indices are >0
        axes_numpy = [ax if ax >= 0 else ndims + ax for ax in axes]

    # Need the fast axis for R2C (last for 'C' order, first for 'F')
    fast_axis = np.argmin(d0.strides)

    if r2c:
        if fast_axis not in axes_numpy:
            raise RuntimeError("The fast axis must be transformed for R2C")

        # For R2C, we need the same fast axis as on the GPU, or the
        # half-hermitian result won't look the same
        if fast_axis != axes_numpy[-1]:
            axes_numpy.remove(fast_axis)
            axes_numpy.append(fast_axis)
        # if order != 'C':
        #     print("R2C", ndims, ndim, shape, axes, axes_numpy, fast_axis, inplace)

    # base FFT scale for numpy (not used for DCT/DST)
    s = np.sqrt(np.prod([d0.shape[i] for i in axes_numpy]))
    if r2c and inplace:
        s = np.sqrt(s ** 2 / d0.shape[fast_axis] * (d0.shape[fast_axis] - r2c_inplace_pad))

    # Tolerance estimated from accuracy notebook
    if dtype in (np.complex64, np.float32):
        tol = 2e-6 + 5e-7 * np.log10(s ** 2)
    else:
        tol = 5e-15 + 5e-16 * np.log10(s ** 2)

    n = max(shape)
    bluestein = max(primes(n)) > 13
    if bluestein:
        tol *= 2
    ############################################################
    # FFT
    ############################################################
    if inplace:
        d1_gpu = d_gpu
    else:
        if r2c:
            if backend == "pyopencl":
                d1_gpu = cla.empty(queue, shapec, dtype=dtype, order=order)
            elif backend == "pycuda":
                d1_gpu = cua.empty(shapec, dtype=dtype, order=order)
            elif backend == "cupy":
                d1_gpu = cp.empty(shapec, dtype=dtype, order=order)
        else:
            # Note that pycuda's gpuarray.copy (as of 2022.2.2) does not copy strides
            d1_gpu = empty_like(d_gpu)

    if has_scipy and ref_long_double:
        # Use long double precision
        if r2c or dct or dst:
            d0n = d0.astype(np.longdouble)
        else:
            d0n = d0.astype(np.clongdouble)
    else:
        d0n = d0

    d1_gpu = app.fft(d_gpu, d1_gpu)
    if not (dct or dst):
        d1_gpu *= app.get_fft_scale()

    if r2c:
        if inplace:
            # Need to cut the fastest axis by 1 or 2
            d = rfftn(np.take(d0n, range(d0n.shape[fast_axis] - r2c_inplace_pad),
                              axis=fast_axis), axes=axes_numpy) / s
        else:
            d = rfftn(d0n, axes=axes_numpy) / s
    elif dct:
        d = dctn(d0n, axes=axes_numpy, type=dct)
    elif dst:
        d = dstn(d0n, axes=axes_numpy, type=dst)
    else:
        d = fftn(d0n, axes=axes_numpy) / s

    if inplace and r2c:
        assert d1_gpu.dtype == dtype, "The array type is incorrect after an inplace FFT"

    n2, ni = l2(d, d1_gpu.get()), li(d, d1_gpu.get())

    src_unchanged_fft = np.all(np.equal(d_gpu.get(), d0))

    # Output string
    if r2c:
        t = "R2C"
    elif dct:
        t = "DCT%d" % dct
    elif dst:
        t = "DST%d" % dst
    else:
        t = "C2C"
    # if r2c and inplace:
    #     tmp = list(d0.shape)
    #     if order == 'C':
    #         tmp[-1] -= r2c_inplace_pad
    #         shstr = str(tuple(tmp)).replace(" ", "")
    #         if ",)" in shstr:
    #             shstr = shstr.replace(",)", f"+{r2c_inplace_pad})")
    #         else:
    #             shstr = shstr.replace(")", f"+{r2c_inplace_pad})")
    #     else:
    #         tmp[0] -= r2c_inplace_pad
    #         if len(tmp) == 1:
    #             shstr = f"({tmp[0]}+{r2c_inplace_pad}),"
    #         else:
    #             shstr = f"({tmp[0]}+{r2c_inplace_pad},"
    #             shstr += str(tuple(tmp[1:])).replace(" ", "").replace(",)", ")")[1:]
    # else:
    #     shstr = str(d0.shape).replace(" ", "")
    shstr = app.get_shape_str()
    shax = str(axes).replace(" ", "").replace("[", "").replace("]", "")
    if colour_output:
        red = max(0, min(int((ni / tol - 0.2) * 255), 255))
        stol = "\x1b[48;2;%d;0;0m%5.1e < %5.1e (%5.3f)\x1b[0m" % (red, ni, tol, ni / tol)
    else:
        stol = "%5.1e < %5.1e (%5.3f)" % (ni, tol, ni / tol)
    nupstr = ''.join(str(nup) for nup in app.get_nb_upload())
    verb_out = "%8s %4s %15s axes=%12s ndim=%4s %5s %5s %10s lut=%4s inplace=%d " \
               " norm=%4s %s %5s: n2=%5.1e ninf=%s %d" % \
               (backend, t, shstr, shax, str(ndim), app.get_algo_str(), nupstr, str(d0.dtype),
                str(use_lut), int(inplace), str(norm), order, "FFT", n2, stol, src_unchanged_fft)

    t3 = timeit.default_timer()

    # Clean memory
    del d_gpu, d1_gpu
    if backend == "cupy" and has_cupy:
        mempool = cp.get_default_memory_pool()
        if mempool is not None:  # Is that test necessary ?
            # Clean memory pool, we are changing array sizes constantly, and using
            # N parallel process so memory management  must be done manually
            mempool.free_all_blocks()

    ############################################################
    # IFFT - from original array to avoid error propagation
    ############################################################
    if r2c:
        # Exception: we need a proper half-Hermitian array
        d0 = d.astype(dtype)
        if has_scipy and ref_long_double:
            d0n = d0.astype(np.clongdouble)
        else:
            d0n = d0
        if (np.argmin(d0.strides) != d0.ndim - 1) and order == 'C':
            # np.fft.rfftn can change the fast axis
            d0 = np.asarray(d0, order='C')
    if order != 'C':
        d0 = np.asarray(d0, order=order)
    if 'opencl' in backend:
        d_gpu = cla.to_device(queue, d0)
    else:
        d_gpu = to_gpu(d0)

    if inplace:
        d1_gpu = d_gpu
    else:
        if r2c:
            if backend == "pyopencl":
                d1_gpu = cla.empty(queue, shape, dtype=dtypef, order=order)
            elif backend == "pycuda":
                d1_gpu = cua.empty(shape, dtype=dtypef, order=order)
            elif backend == "cupy":
                d1_gpu = cp.empty(shape, dtype=dtypef, order=order)
        else:
            d1_gpu = empty_like(d_gpu)

    d1_gpu = app.ifft(d_gpu, d1_gpu)
    if not (dct or dst):
        d1_gpu *= app.get_ifft_scale()

    if r2c:
        # We need to specify the destination shape, in the case
        # we want an odd-sized fast axis
        d = irfftn(d0n, s=[shape0[i] for i in axes_numpy], axes=axes_numpy) * s
    elif dct:
        d = idctn(d0n, axes=axes_numpy, type=dct)
    elif dst:
        d = idstn(d0n, axes=axes_numpy, type=dst)
    else:
        d = ifftn(d0n, axes=axes_numpy) * s

    if inplace:
        if dct or dst or r2c:
            assert d1_gpu.dtype == dtypef, "The array type is incorrect after an inplace iFFT"
        else:
            assert d1_gpu.dtype == dtype, "The array type is incorrect after an inplace iFFT"

    if r2c and inplace:
        tmp = np.take(d1_gpu.get(), range(d1_gpu.shape[fast_axis] - r2c_inplace_pad), axis=fast_axis)
        n2i, nii = l2(d, tmp), li(d, tmp)
    else:
        n2i, nii = l2(d, d1_gpu.get()), li(d, d1_gpu.get())

    src_unchanged_ifft = np.all(np.equal(d_gpu.get(), d0))

    # Max N for radix 1D C2R transforms to not overwrite source
    if platform.system() == 'Darwin':
        nmaxr2c1d = 2048 * (1 + int(dtype in (np.float32, np.complex64)))
    else:
        nmaxr2c1d = 3072 * (1 + int(dtype in (np.float32, np.complex64)))
    ndimtmp = ndim if ndim is not None else len(axes)
    if max(ni, nii) <= tol and (inplace or src_unchanged_fft) and \
            (inplace or src_unchanged_ifft or (r2c and ndimtmp > 1 or n >= nmaxr2c1d or bluestein)):
        success = 'OK'
    else:
        success = 'FAIL'

    if colour_output:
        red = max(0, min(int((nii / tol - 0.2) * 255), 255))
        stol = "\x1b[48;2;%d;0;0m%5.1e < %5.1e (%5.3f)\x1b[0m" % (red, nii, tol, nii / tol)
    else:
        stol = "%5.1e < %5.1e (%5.3f)" % (nii, tol, nii / tol)
    verb_out += "%5s: n2=%5.1e ninf=%s %d" % ("iFFT", n2i, stol, src_unchanged_ifft)

    # Also print the size of the allocated buffer and success
    verb_out += f" buf={app.get_tmp_buffer_str()} {success:4s}"

    if verbose:
        print(verb_out)

    t4 = timeit.default_timer()

    if backend == "pyopencl":
        gpu_name = gpu_ctx_dic["pyopencl"][0].name
    elif backend == "pycuda":
        gpu_name = gpu_ctx_dic["pycuda"][0].name()
    else:
        gpu_name = ""

    res = {"n2": n2, "ni": ni, "n2i": n2i, "nii": nii, "tol": tol, "dt_array": t1 - t0, "dt_app": t2 - t1,
           "dt_fft": t3 - t2, "dt_ifft": t4 - t3, "src_unchanged_fft": src_unchanged_fft,
           "src_unchanged_ifft": src_unchanged_ifft, "tol_test": max(ni, nii) < tol, "str": verb_out,
           "backend": backend, "shape": shape0, "ndim": ndim, "axes": axes, "dtype": dtype0, "inplace": inplace,
           "norm": norm, "use_lut": use_lut, "r2c": r2c, "dct": dct, "dst": dst,
           "gpu_name": gpu_name, "order": order}

    if return_array:
        res["d0"] = d0
    return res


def test_accuracy_kwargs(kwargs):
    # This function must be defined here, so it can be used with a multiprocessing pool
    # in test_fft, otherwise this will fail, see:
    # https://stackoverflow.com/questions/41385708/multiprocessing-example-giving-attributeerror
    if kwargs['backend'] == 'pyopencl' and has_opencl:
        try:
            t = test_accuracy(**kwargs)
        except cl.RuntimeError as ex:
            # The cl.RuntimeError can't be pickled, so is not correctly reported
            # when using multiprocessing. So we raise another and the traceback
            # from the previous one is still printed.
            raise RuntimeError("An OpenCL RuntimeError was encountered")
        return t
    return test_accuracy(**kwargs)


def exhaustive_test(backend, vn, ndim, dtype, inplace, norm, use_lut, r2c=False,
                    dct=False, dst=False, nproc=None,
                    verbose=True, return_res=False):
    """
    Run tests on a large range of sizes using multiprocessing. Manual function.

    :param backend: either 'pyopencl', 'pycuda' or 'cupy'
    :param vn: the list/iterable of sizes n.
    :param ndim: the number of dimensions. The array shape will be [n]*ndim
    :param dtype: either np.complex64 or np.complex128, or np.float32/np.float64 for r2c/dct/dst
    :param inplace: True or False
    :param norm: either 0, 1 or "ortho"
    :param use_lut: if True,1, False or 0, will trigger useLUT=1 or 0 for VkFFT.
        If None, the default VkFFT behaviour is used. Always True by default
        for double precision, so no need to force it.
    :param r2c: if True, test an r2c transform. If inplace, the last dimension
        (x, fastest axis) must be even
    :param dct: either 1, 2, 3 or 4 to test different dct. Only norm=1 is can be
        tested (native scipy normalisation).
    :param dst: either 1, 2, 3 or 4 to test different dst. Only norm=1 is can be
        tested (native scipy normalisation).
    :param nproc: the maximum number of parallel process to use. If None, the
        number of detected cores will be used (this may use too much memory !)
    :param verbose: if True, prints 1 line per test
    :param return_res: if True, return the list of result dictionaries.
    :return: True if all tests passed, False otherwise. If return_res is True, return
        the list of result dictionaries instead.
    """
    try:
        # Get the real number of processor cores available
        # os.sched_getaffinity is only available on some *nix platforms
        nproc1 = len(os.sched_getaffinity(0)) * psutil.cpu_count(logical=False) // psutil.cpu_count(logical=True)
    except AttributeError:
        nproc1 = os.cpu_count()
    if nproc is None:
        nproc = nproc1
    else:
        nproc = min(nproc, nproc1)
    # Generate the list of configurations as kwargs for test_accuracy()
    vkwargs = []
    for n in vn:
        kwargs = {"backend": backend, "shape": [n] * ndim, "ndim": ndim, "axes": None, "dtype": dtype,
                  "inplace": inplace, "norm": norm, "use_lut": use_lut,
                  "r2c": r2c, "dct": dct, "dst": dst, "stream": None,
                  "verbose": False}
        vkwargs.append(kwargs)
    vok = []
    vres = []
    # Need to use spawn to handle the GPU context
    with multiprocessing.get_context('spawn').Pool(nproc) as pool:
        for res in pool.imap(test_accuracy_kwargs, vkwargs):
            # TODO: this should better be logged
            if verbose:
                print(res['str'])
            ni, n2 = res["ni"], res["n2"]
            nii, n2i = res["nii"], res["n2i"]
            tol = res["tol"]
            ok = max(ni, nii) < tol
            if not inplace:
                ok = ok and res["src_unchanged_fft"]
                if not r2c:
                    ok = ok and res["src_unchanged_ifft"]
            vok.append(ok)
            if return_res:
                vres.append(res)
    if return_res:
        return vres
    return np.alltrue(vok)
