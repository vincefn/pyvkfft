# -*- coding: utf-8 -*-

# PyVkFFT
#   (c) 2021- : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr
#

"""
Benchmark functions. These are implemented using separate process,
one for each test - this involves a fair amount of overhead, but avoids
any resource conflict, or issue with GPU contexts, deletion of cufft
plans (https://github.com/lebedov/scikit-cuda/issues/308), etc..
"""

import warnings
import os
import timeit
from multiprocessing import Process, Queue
import numpy as np


# test for GPU packages in parallel process (slower but cleaner)


def _test_pyvkfft_cuda(q):
    try:
        import pycuda.autoprimaryctx
        import pycuda.driver as cuda
        import pycuda.gpuarray as cua
        from pycuda import curandom
        import pyvkfft.cuda
        from pyvkfft.cuda import primes, VkFFTApp as cuVkFFTApp
        q.put(True)
    except:
        q.put(False)


def test_pyvkfft_cuda():
    """
    Test if pyvkfft_cuda is available. The test is made in a separate process.
    """
    q = Queue()
    p = Process(target=_test_pyvkfft_cuda, args=(q,))
    p.start()
    has_pyvkfft_cuda = q.get()
    p.join()
    return has_pyvkfft_cuda


def _test_pyvkfft_opencl(q):
    try:
        import pyopencl as cl
        import pyopencl.array as cla
        from pyopencl import clrandom
        import pyvkfft.opencl
        from pyvkfft.opencl import primes, VkFFTApp as clVkFFTApp
        q.put(True)
    except:
        q.put(False)


def test_pyvkfft_opencl():
    """
    Test if pyvkfft_opencl is available. The test is made in a separate process.
    """
    q = Queue()
    p = Process(target=_test_pyvkfft_opencl, args=(q,))
    p.start()
    has_pyvkfft_opencl = q.get()
    p.join()
    return has_pyvkfft_opencl


def _test_skcuda(q):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            import pycuda.autoprimaryctx
            import pycuda.driver as cu_drv
            import pycuda.gpuarray as cua
            from pycuda import curandom
            import skcuda.fft as cu_fft
            q.put(True)
        except:
            q.put(False)


def test_skcuda():
    q = Queue()
    p = Process(target=_test_skcuda, args=(q,))
    p.start()
    has_skcuda = q.get()
    p.join()
    return has_skcuda


def _test_gpyfft(q):
    """
    Test if scikit-cuda is available. The test is made in a separate process.
    """
    try:
        import pyopencl as cl
        import pyopencl.array as cla
        from pyopencl import clrandom
        import gpyfft
        q.put(True)
    except:
        q.put(False)


def test_gpyfft():
    """
    Test if gpyfft is available. The test is made in a separate process.
    """
    q = Queue()
    p = Process(target=test_gpyfft, args=(q,))
    p.start()
    has_gpyfft = q.get()
    p.join()
    return has_gpyfft


def _bench_pyvkfft_opencl(q, sh, precision='single', ndim=1, nb_repeat=3, gpu_name=None, opencl_platform=None):
    import pyopencl as cl
    import pyopencl.array as cla
    from pyopencl import clrandom
    import pyvkfft.opencl
    from pyvkfft.opencl import primes, VkFFTApp as clVkFFTApp
    dtype = np.complex128 if precision == 'double' else np.complex64
    if 'PYOPENCL_CTX' in os.environ:
        cl_ctx = cl.create_some_context()
    else:
        cl_ctx = None
        for p in cl.get_platforms():
            if opencl_platform is not None:
                if opencl_platform.lower() not in p.name.lower():
                    continue
            for d in p.get_devices():
                if gpu_name is not None:
                    if gpu_name.lower() not in d.name.lower():
                        continue
                if d.type & cl.device_type.GPU == 0:
                    continue
                gpu_name_real = "%s:%s" % (p.name, d.name)
                # print("Selected OpenCL device: ", d.name)
                cl_ctx = cl.Context(devices=(d,))
                break
            if cl_ctx is not None:
                break
    cq = cl.CommandQueue(cl_ctx)
    dt = 0
    d = clrandom.rand(cq, shape=sh, dtype=np.float32).astype(dtype)
    if True:
        app = clVkFFTApp(d.shape, dtype=dtype, queue=cq, ndim=ndim)
        for i in range(nb_repeat):
            cq.finish()
            t0 = timeit.default_timer()
            d = app.ifft(d)
            d = app.fft(d)
            cq.finish()
            dt1 = timeit.default_timer() - t0
            if dt == 0:
                dt = dt1
            elif dt1 < dt:
                dt = dt1
        # print("%4d %4dx%4d 2D FFT+iFFT dt=%6.2f ms %7.2f Gbytes/s [pyvkfft.opencl]  [nb=%4d]" %
        #      (nz, n, n, dt / nb * 1000, gbps, nb))
        gbps = d.nbytes * ndim * 2 * 2 / dt / 1024 ** 3
    else:
        gbps = 0
    q.put((dt, gbps, gpu_name_real))


def bench_pyvkfft_opencl(sh, precision='single', ndim=1, nb_repeat=3, gpu_name=None, opencl_platform=None):
    q = Queue()
    p = Process(target=_bench_pyvkfft_opencl, args=(q, sh, precision, ndim, nb_repeat, gpu_name))
    p.start()
    dt, gbps, gpu_name_real = q.get()
    p.join()
    return dt, gbps, gpu_name_real


def _bench_pyvkfft_cuda(q, sh, precision='single', ndim=1, nb_repeat=3, gpu_name=None):
    import pycuda.autoprimaryctx  # See https://github.com/lebedov/scikit-cuda/issues/330#issuecomment-1125471345
    import pycuda.driver as cu_drv
    import pycuda.gpuarray as cua
    from pycuda import curandom
    import pyvkfft.cuda
    from pyvkfft.cuda import primes, VkFFTApp as cuVkFFTApp
    dtype = np.complex128 if precision == 'double' else np.complex64
    if gpu_name is None:
        d = cu_drv.Device(0)
        gpu_name_real = d.name()
        # print("Selected  CUDA  device: ", d.name())
        cu_ctx = d.make_context()
    else:
        for i in range(cu_drv.Device.count()):
            d = cu_drv.Device(i)
            if gpu_name.lower() in d.name().lower():
                gpu_name_real = d.name()
                # print("Selected  CUDA  device: ", d.name())
                cu_ctx = d.make_context()
                break
    dt = 0
    d = curandom.rand(shape=sh, dtype=np.float32).astype(dtype)
    try:
        app = cuVkFFTApp(d.shape, dtype=dtype, ndim=ndim)
        for i in range(nb_repeat):
            cu_ctx.synchronize()
            t0 = timeit.default_timer()
            d = app.ifft(d)
            d = app.fft(d)
            cu_ctx.synchronize()
            dt1 = timeit.default_timer() - t0
            if dt == 0:
                dt = dt1
            elif dt1 < dt:
                dt = dt1
        # print("%4d %4dx%4d 2D FFT+iFFT dt=%6.2f ms %7.2f Gbytes/s [pyvkfft.cuda]    [nb=%4d]" %
        #      (nz, n, n, dt / nb * 1000, gbps, nb))
        del app
        gbps = d.nbytes * ndim * 2 * 2 / dt / 1024 ** 3
    except:
        gbps = 0
    q.put((dt, gbps, gpu_name_real))


def bench_pyvkfft_cuda(sh, precision='single', ndim=1, nb_repeat=3, gpu_name=None):
    q = Queue()
    p = Process(target=_bench_pyvkfft_cuda, args=(q, sh, precision, ndim, nb_repeat, gpu_name))
    p.start()
    dt, gbps, gpu_name_real = q.get(timeout=5)
    p.join()
    return dt, gbps, gpu_name_real


def _bench_skcuda(q, sh, precision='single', ndim=1, nb_repeat=3, gpu_name=None):
    import pycuda.autoprimaryctx  # See https://github.com/lebedov/scikit-cuda/issues/330#issuecomment-1125471345
    import pycuda.driver as cu_drv
    from pycuda import curandom
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import skcuda.fft as cu_fft
    dtype = np.complex128 if precision == 'double' else np.complex64
    if gpu_name is None:
        d = cu_drv.Device(0)
        gpu_name_real = d.name()
        cu_ctx = d.make_context()
    else:
        for i in range(cu_drv.Device.count()):
            d = cu_drv.Device(i)
            if gpu_name.lower() in d.name().lower():
                gpu_name_real = d.name()
                cu_ctx = d.make_context()
                break
    dt = 0
    d = curandom.rand(shape=sh, dtype=np.float32).astype(dtype)
    if ndim == 1:
        plan = cu_fft.Plan(sh[-1], dtype, dtype, batch=sh[-2])
    elif ndim == 2:
        plan = cu_fft.Plan(sh[-2:], dtype, dtype, batch=sh[-3])
    else:
        plan = cu_fft.Plan(sh[-3:], dtype, dtype, batch=sh[-4])
    dt = 0
    for i in range(nb_repeat):
        cu_ctx.synchronize()
        t0 = timeit.default_timer()
        cu_fft.fft(d, d, plan)
        cu_fft.ifft(d, d, plan)
        cu_ctx.synchronize()
        dt1 = timeit.default_timer() - t0
        if dt == 0:
            dt = dt1
        elif dt1 < dt:
            dt = dt1
    gbps = d.nbytes * ndim * 2 * 2 / dt / 1024 ** 3
    q.put((dt, gbps, gpu_name_real))


def bench_skcuda(sh, precision='single', ndim=1, nb_repeat=3, gpu_name=None):
    q = Queue()
    p = Process(target=_bench_skcuda, args=(q, sh, precision, ndim, nb_repeat, gpu_name))
    p.start()
    dt, gbps, gpu_name_real = q.get(timeout=5)
    p.join()
    return dt, gbps, gpu_name_real
