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
import platform
from itertools import permutations
from time import localtime, strftime
import numpy as np
import matplotlib.pyplot as plt
from pyvkfft.version import vkfft_version, vkfft_git_version
from pyvkfft.base import primes


# test for GPU packages in parallel process (slower but cleaner)


def _test_pyvkfft_cuda(q):
    try:
        import pycuda.autoinit
        import pycuda.driver as cuda
        import pycuda.gpuarray as cua
        from pycuda import curandom
        import pyvkfft.cuda
        from pyvkfft.cuda import VkFFTApp as cuVkFFTApp, cuda_compile_version, \
            cuda_driver_version, cuda_runtime_version
        q.put((True, cuda_compile_version(), cuda_driver_version(), cuda_runtime_version()))
    except:
        q.put((False, None, None, None))


def test_pyvkfft_cuda():
    """
    Test if pyvkfft_cuda is available. The test is made in a separate process.
    Also return the
    """
    q = Queue()
    p = Process(target=_test_pyvkfft_cuda, args=(q,))
    p.start()
    has_pyvkfft_cuda, cu_version_compile, cu_version_driver, cu_version_runtime = q.get()
    p.join()
    return has_pyvkfft_cuda, cu_version_compile, cu_version_driver, cu_version_runtime


def _test_pyvkfft_opencl(q):
    try:
        import pyopencl as cl
        import pyopencl.array as cla
        from pyopencl import clrandom
        import pyvkfft.opencl
        from pyvkfft.opencl import VkFFTApp as clVkFFTApp
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
            import pycuda.autoinit
            import pycuda.driver as cu_drv
            import pycuda.gpuarray as cua
            from pycuda import curandom
            import skcuda.fft as cu_fft
            from skcuda.cufft import _cufft_version as v
            q.put((True, "%d.%d.%d" % (v // 1000, (v // 100) % 100, v % 100)))
        except:
            q.put((False, None))


def test_skcuda():
    q = Queue()
    p = Process(target=_test_skcuda, args=(q,))
    p.start()
    has_skcuda, cufft_version = q.get()
    p.join()
    return has_skcuda, cufft_version


def _test_cupy(q):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            import cupy as cp
            with cp.cuda.Device(0).use():
                vd = cp.cuda.runtime.driverGetVersion()
                vr = cp.cuda.runtime.runtimeGetVersion()
                q.put((True, "%d.%d.%d" % (vd // 1000, (vd // 100) % 100, vd % 100),
                       "%d.%d.%d" % (vr // 1000, (vr // 100) % 100, vr % 100)))
        except:
            q.put((False, None))


def test_cupy():
    q = Queue()
    p = Process(target=_test_cupy, args=(q,))
    p.start()
    has_cupy, cuda_driver_version, cuda_runtime_version = q.get()
    p.join()
    return has_cupy, cuda_driver_version, cuda_runtime_version


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
    p = Process(target=_test_gpyfft, args=(q,))
    p.start()
    has_gpyfft = q.get()
    p.join()
    return has_gpyfft


def _bench_pyvkfft_opencl(q, sh, precision='single', ndim=1, nb_repeat=3, nb_loop=1, gpu_name=None,
                          opencl_platform=None, args=None, inplace=True, r2c=False, dct=False, dst=False):
    import pyopencl as cl
    import pyopencl.array as cla
    from pyopencl import clrandom
    import pyvkfft.opencl
    from pyvkfft.opencl import VkFFTApp as clVkFFTApp

    if r2c or dct or dst:
        dtype = np.float64 if precision == 'double' else np.float32
    else:
        dtype = np.complex128 if precision == 'double' else np.complex64

    gpu_name_real = gpu_name
    platform_name_real = opencl_platform
    if 'PYOPENCL_CTX' in os.environ:
        cl_ctx = cl.create_some_context()
    else:
        cl_ctx = None
        has_pocl = False
        for p in cl.get_platforms():
            if opencl_platform is not None:
                if opencl_platform.lower() not in p.name.lower():
                    continue
            elif "portable" in p.name.lower():
                # Try to skip PoCL unless it was requested
                has_pocl = True
                continue
            for d in p.get_devices():
                if gpu_name is not None:
                    if gpu_name.lower() not in d.name.lower():
                        continue
                if d.type & cl.device_type.GPU == 0:
                    continue
                gpu_name_real = d.name
                platform_name_real = p.name
                # print("Selected OpenCL device: ", d.name)
                cl_ctx = cl.Context(devices=(d,))
                break
            if cl_ctx is not None:
                break
        if cl_ctx is None and opencl_platform is None and has_pocl:
            # Try again without excluding PoCL
            for p in cl.get_platforms():
                for d in p.get_devices():
                    if gpu_name is not None:
                        if gpu_name.lower() not in d.name.lower():
                            continue
                    if d.type & cl.device_type.GPU == 0:
                        continue
                    gpu_name_real = d.name
                    platform_name_real = p.name
                    # print("Selected OpenCL device: ", d.name)
                    cl_ctx = cl.Context(devices=(d,))
                    break
                if cl_ctx is not None:
                    break
    cq = cl.CommandQueue(cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    dt = 0

    r2c_odd = False
    if r2c and inplace:
        # Pad with 1 or 2 elements for inplace r2c
        sh = list(sh)
        if sh[-1] % 2:
            r2c_odd = True
            sh[-1] += 1
        else:
            sh[-1] += 2
        sh = tuple(sh)

    d = clrandom.rand(cq, shape=sh, dtype=np.float32).astype(dtype)
    if inplace:
        d1 = d
    else:
        if r2c:
            sh1 = [n for n in d.shape]
            sh1[-1] = sh1[-1] // 2 + 1
            d1 = cla.empty(cq, shape=tuple(sh1), dtype=np.complex128 if precision == 'double' else np.complex64)
        else:
            d1 = cla.empty_like(d)
    try:
        kwargs = {}
        if args is not None:
            for k, v in args.items():
                if k in ["disableReorderFourStep", "coalescedMemory", "numSharedBanks",
                         "aimThreads", "performBandwidthBoost", "registerBoost",
                         "registerBoostNonPow2", "registerBoost4Step", "warpSize", "useLUT",
                         "groupedBatch", "forceCallbackVersionRealTransforms", "tune_config"]:
                    kwargs[k] = v
        app = clVkFFTApp(d.shape, dtype=dtype, queue=cq, ndim=ndim, inplace=inplace,
                         r2c=r2c, dct=dct, dst=dst, r2c_odd=r2c_odd, **kwargs)
        algo_str = app.get_algo_str()
        nup_str = ''.join(str(nup) for nup in app.get_nb_upload())
        vkfft_str = f"algo={algo_str} buf={app.get_tmp_buffer_str()} up={nup_str}"
        if 'tune_config' in kwargs:
            for k, v in kwargs['tune_config'].items():
                if k == "backend":
                    continue
                vkfft_str += f" {k}={getattr(app, k)}"
        for i in range(nb_repeat):
            cq.finish()
            t0 = timeit.default_timer()
            # Apparently OpenCL events don't always work. Need kernel events ?
            # start = cl.enqueue_marker(cq)
            for ii in range(nb_loop):
                d1 = app.fft(d, d1)
                d = app.ifft(d1, d)
            # end = cl.enqueue_marker(cq)
            # end.wait()
            # dt1 = 1e-9 * (start.profile.END - end.profile.END)
            cq.finish()
            dt1 = (timeit.default_timer() - t0) / nb_loop
            if dt == 0:
                dt = dt1
            elif dt1 < dt:
                dt = dt1
        # print("%4d %4dx%4d 2D FFT+iFFT dt=%6.2f ms %7.2f Gbytes/s [pyvkfft.opencl]  [nb=%4d]" %
        #      (nz, n, n, dt / nb * 1000, gbps, nb))
        gbps = d.nbytes * ndim * 2 * 2 / dt / 1024 ** 3
    except Exception:
        import traceback
        print(traceback.format_exc())
        gbps = 0
    results = {'dt': dt, 'gbps': gbps, 'gpu_name_real': gpu_name_real,
               'platform_name_real': platform_name_real,
               'vkfft_str': vkfft_str, 'algo_str': algo_str, 'nup_str': nup_str}
    if q is None:
        return results
    else:
        q.put(results)


def bench_pyvkfft_opencl(sh, precision='single', ndim=1, nb_repeat=3, nb_loop=1, gpu_name=None,
                         opencl_platform=None, args=None, serial=False,
                         inplace=True, r2c=False, dct=False, dst=False):
    if serial:
        return _bench_pyvkfft_opencl(None, sh, precision, ndim, nb_repeat, nb_loop, gpu_name,
                                     opencl_platform, args, inplace=inplace,
                                     r2c=r2c, dct=dct, dst=dst)
    q = Queue()
    p = Process(target=_bench_pyvkfft_opencl, args=(q, sh, precision, ndim, nb_repeat, nb_loop, gpu_name,
                                                    opencl_platform, args, inplace, r2c, dct, dst))
    p.start()
    try:
        results = q.get(timeout=20)
    except:
        results = {'dt': 0, 'gbps': 0, 'gpu_name_real': None, 'platform_name_real': None,
                   'vkfft_str': None}
    p.join()
    return results


def _bench_pyvkfft_cuda(q, sh, precision='single', ndim=1, nb_repeat=3, nb_loop=1, gpu_name=None, args=None,
                        inplace=True, r2c=False, dct=False, dst=False):
    import pycuda.autoprimaryctx  # See https://github.com/lebedov/scikit-cuda/issues/330#issuecomment-1125471345
    import pycuda.driver as cu_drv
    import pycuda.gpuarray as cua
    from pycuda import curandom
    import pyvkfft.cuda
    from pyvkfft.cuda import VkFFTApp as cuVkFFTApp

    if r2c or dct or dst:
        dtype = np.float64 if precision == 'double' else np.float32
    else:
        dtype = np.complex128 if precision == 'double' else np.complex64

    if gpu_name is None:
        gpu_name_real = pycuda.autoprimaryctx.device.name()
        cu_ctx = pycuda.autoprimaryctx.context
    else:
        for i in range(cu_drv.Device.count()):
            d = cu_drv.Device(i)
            if gpu_name.lower() in d.name().lower():
                gpu_name_real = d.name()
                cu_ctx = d.retain_primary_context()
                break
    cu_ctx.push()
    dt = 0

    r2c_odd = False
    if r2c and inplace:
        # Pad with 1 or 2 elements for inplace r2c
        sh = list(sh)
        if sh[-1] % 2:
            r2c_odd = True
            sh[-1] += 1
        else:
            sh[-1] += 2
        sh = tuple(sh)

    d = curandom.rand(shape=sh, dtype=np.float32).astype(dtype)

    if inplace:
        d1 = d
    else:
        if r2c:
            sh1 = [n for n in d.shape]
            sh1[-1] = sh1[-1] // 2 + 1
            d1 = cua.empty(shape=tuple(sh1), dtype=np.complex128 if precision == 'double' else np.complex64)
        else:
            d1 = cua.empty_like(d)

    try:
        kwargs = {}
        if args is not None:
            for k, v in args.items():
                if k in ["disableReorderFourStep", "coalescedMemory", "numSharedBanks",
                         "aimThreads", "performBandwidthBoost", "registerBoost",
                         "registerBoostNonPow2", "registerBoost4Step", "warpSize", "useLUT",
                         "groupedBatch", "tune_config"]:
                    kwargs[k] = v
        app = cuVkFFTApp(d.shape, dtype=dtype, ndim=ndim, inplace=inplace,
                         r2c=r2c, dct=dct, dst=dst, r2c_odd=r2c_odd, **kwargs)
        algo_str = app.get_algo_str()
        nup_str = ''.join(str(nup) for nup in app.get_nb_upload())
        vkfft_str = f"algo={algo_str} buf={app.get_tmp_buffer_str()} up={nup_str}"
        if 'tune_config' in kwargs:
            for k, v in kwargs['tune_config'].items():
                if k == "backend":
                    continue
                vkfft_str += f" {k}={getattr(app, k)}"
        start = cu_drv.Event()
        stop = cu_drv.Event()
        for i in range(nb_repeat):
            cu_ctx.synchronize()
            start.record()
            for ii in range(nb_loop):
                d1 = app.fft(d, d1)
                d = app.ifft(d1, d)
            stop.record()
            cu_ctx.synchronize()
            dt1 = stop.time_since(start) / 1000 / nb_loop
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
    cu_ctx.pop()
    results = {'dt': dt, 'gbps': gbps, 'gpu_name_real': gpu_name_real, 'vkfft_str': vkfft_str,
               'algo_str': algo_str, 'nup_str': nup_str}
    if q is None:
        return results
    else:
        q.put(results)


def bench_pyvkfft_cuda(sh, precision='single', ndim=1, nb_repeat=3, nb_loop=1, gpu_name=None, args=None, serial=False,
                       inplace=True, r2c=False, dct=False, dst=False):
    if serial:
        return _bench_pyvkfft_cuda(None, sh, precision, ndim, nb_repeat, nb_loop, gpu_name, args, inplace=inplace,
                                   r2c=r2c, dct=dct, dst=dst)
    q = Queue()
    p = Process(target=_bench_pyvkfft_cuda, args=(q, sh, precision, ndim, nb_repeat, nb_loop, gpu_name,
                                                  args, inplace, r2c, dct, dst))
    p.start()
    try:
        results = q.get(timeout=20)
    except:
        results = {'dt': 0, 'gbps': 0, 'gpu_name_real': None, 'vkfft_str': None}
    p.join()
    return results


def _bench_cupy(q, sh, precision='single', ndim=1, nb_repeat=3, nb_loop=1, gpu_name=None):
    import cupy as cp
    import cupyx
    dtype = np.complex128 if precision == 'double' else np.complex64
    if gpu_name is None:
        dev = cp.cuda.Device(0).use()
    else:
        for i in range(cp.cuda.runtime.getDeviceCount()):
            if gpu_name.lower() in cp.cuda.runtime.getDeviceProperties(i)['name'].decode().lower():
                dev = cp.cuda.Device(i).use()
                break
    gpu_name_real = cp.cuda.runtime.getDeviceProperties(dev.id)['name'].decode()
    d = cp.random.uniform(0, 1, sh, dtype=np.float32).astype(dtype)
    dt = 0
    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    ax = list(range(len(sh)))[-ndim:]
    # Explicitly creating the plan does not speed up
    # plan = cupyx.scipy.fft.get_fft_plan(d, axes=ax, value_type='C2C')
    for i in range(nb_repeat):
        dev.synchronize()
        start.record()
        for ii in range(nb_loop):
            cupyx.scipy.fft.fftn(d, axes=ax, overwrite_x=True)
            cupyx.scipy.fft.ifftn(d, axes=ax, overwrite_x=True)
        # a = cp.fft.fftn(d, axes=ax)
        # a = cp.fft.ifftn(d, axes=ax)
        stop.record()
        dev.synchronize()
        dt1 = cp.cuda.get_elapsed_time(start, stop) / 1000 / nb_loop
        if dt == 0:
            dt = dt1
        elif dt1 < dt:
            dt = dt1
    gbps = d.nbytes * ndim * 2 * 2 / dt / 1024 ** 3
    results = {'dt': dt, 'gbps': gbps, 'gpu_name_real': gpu_name_real}
    if q is None:
        return results
    else:
        q.put(results)


def bench_cupy(sh, precision='single', ndim=1, nb_repeat=3, nb_loop=1, gpu_name=None, serial=False):
    if serial:
        return _bench_cupy(None, sh, precision, ndim, nb_repeat, nb_loop, gpu_name)
    q = Queue()
    p = Process(target=_bench_cupy, args=(q, sh, precision, ndim, nb_repeat, nb_loop, gpu_name))
    p.start()
    try:
        results = q.get(timeout=10)
    except:
        results = {'dt': 0, 'gbps': 0, 'gpu_name_real': None}
    p.join()
    return results


def _bench_skcuda(q, sh, precision='single', ndim=1, nb_repeat=3, nb_loop=1, gpu_name=None):
    import pycuda.autoprimaryctx  # See https://github.com/lebedov/scikit-cuda/issues/330#issuecomment-1125471345
    import pycuda.driver as cu_drv
    from pycuda import curandom
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import skcuda.fft as cu_fft
    dtype = np.complex128 if precision == 'double' else np.complex64
    if gpu_name is None:
        gpu_name_real = pycuda.autoprimaryctx.device.name()
        cu_ctx = pycuda.autoprimaryctx.context
    else:
        for i in range(cu_drv.Device.count()):
            d = cu_drv.Device(i)
            if gpu_name.lower() in d.name().lower():
                gpu_name_real = d.name()
                cu_ctx = d.retain_primary_context()
                break
    cu_ctx.push()
    d = curandom.rand(shape=sh, dtype=np.float32).astype(dtype)
    if ndim == 1:
        plan = cu_fft.Plan(sh[-1], dtype, dtype, batch=sh[-2])
    elif ndim == 2:
        plan = cu_fft.Plan(sh[-2:], dtype, dtype, batch=sh[-3])
    else:
        plan = cu_fft.Plan(sh[-3:], dtype, dtype, batch=sh[-4])
    dt = 0
    start = cu_drv.Event()
    stop = cu_drv.Event()
    for i in range(nb_repeat):
        cu_ctx.synchronize()
        start.record()
        for ii in range(nb_loop):
            cu_fft.fft(d, d, plan)
            cu_fft.ifft(d, d, plan)
        stop.record()
        cu_ctx.synchronize()
        dt1 = stop.time_since(start) / 1000 / nb_loop
        if dt == 0:
            dt = dt1
        elif dt1 < dt:
            dt = dt1
    gbps = d.nbytes * ndim * 2 * 2 / dt / 1024 ** 3
    cu_ctx.pop()
    results = {'dt': dt, 'gbps': gbps, 'gpu_name_real': gpu_name_real}
    if q is None:
        return results
    else:
        q.put(results)


def bench_skcuda(sh, precision='single', ndim=1, nb_repeat=3, nb_loop=1, gpu_name=None, serial=False):
    if serial:
        return _bench_skcuda(None, sh, precision, ndim, nb_repeat, nb_loop, gpu_name)
    q = Queue()
    p = Process(target=_bench_skcuda, args=(q, sh, precision, ndim, nb_repeat, nb_loop, gpu_name))
    p.start()
    try:
        results = q.get(timeout=10)
    except:
        results = {'dt': 0, 'gbps': 0, 'gpu_name_real': None}
    p.join()
    return results


def _bench_gpyfft(q, sh, precision='single', ndim=1, nb_repeat=3, nb_loop=1, gpu_name=None, opencl_platform=None):
    results = {'dt': 0, 'gbps': 0, 'gpu_name_real': None, 'platform_name_real': None}
    if max(primes(sh[-1])) > 13:
        q.put(results)
    else:
        import pyopencl as cl
        from pyopencl import clrandom
        import gpyfft
        dtype = np.complex128 if precision == 'double' else np.complex64
        gpu_name_real = gpu_name
        platform_name_real = opencl_platform
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
                    gpu_name_real = d.name
                    platform_name_real = p.name
                    # print("Selected OpenCL device: ", d.name)
                    cl_ctx = cl.Context(devices=(d,))
                    break
                if cl_ctx is not None:
                    break
        cq = cl.CommandQueue(cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
        dt = 0
        d = clrandom.rand(cq, shape=sh, dtype=np.float32).astype(dtype)
        for axes in permutations([-1, -2, -3][:ndim]):
            # Shuffle axes order to find fastest transform
            gpyfft_plan = gpyfft.FFT(cl_ctx, cq, d, None, axes=axes)
            for i in range(nb_repeat):
                cq.finish()
                t0 = timeit.default_timer()
                for ii in range(nb_loop):
                    gpyfft_plan.enqueue(forward=True)
                    gpyfft_plan.enqueue(forward=False)
                cq.finish()
                dt1 = (timeit.default_timer() - t0) / nb_loop
                if dt == 0:
                    dt = dt1
                elif dt1 < dt:
                    dt = dt1
            del gpyfft_plan
        gbps = d.nbytes * ndim * 2 * 2 / dt / 1024 ** 3
        results = {'dt': dt, 'gbps': gbps, 'gpu_name_real': gpu_name_real,
                   'platform_name_real': platform_name_real}
        if q is None:
            return results
        else:
            q.put(results)


def bench_gpyfft(sh, precision='single', ndim=1, nb_repeat=3, nb_loop=1, gpu_name=None, opencl_platform=None,
                 serial=False):
    if serial:
        return _bench_gpyfft(None, sh, precision, ndim, nb_repeat, nb_loop, gpu_name, opencl_platform)
    q = Queue()
    p = Process(target=_bench_gpyfft, args=(q, sh, precision, ndim, nb_repeat, nb_loop, gpu_name, opencl_platform))
    p.start()
    try:
        results = q.get(timeout=10)
    except:
        results = {'dt': 0, 'gbps': 0, 'gpu_name_real': None, 'platform_name_real': None}
    p.join()
    return results


def plot_benchmark(results, ndim, gpu_name_real, radix_max, legend_loc="lower left",
                   fig=None, figsize=(16, 8)):
    if fig is None:
        plt.figure(figsize=figsize)
    else:
        plt.clf()
    if radix_max > 13:
        # Use a different symbol for Bluestein
        x = results['n']
        maxprime = np.array(results['maxprime'])
        idx7a = np.where(maxprime <= 7)[0]
        idx7b = np.where(maxprime > 7)[0]
        idx13a = np.where(maxprime <= 13)[0]
        idx13b = np.where(maxprime > 13)[0]
        if "gpyfft[clFFT]" in results:
            y = results["gpyfft[clFFT]"]
            plt.plot(np.take(x, idx13a), np.take(y, idx13a), color='#00FF0D', marker='o', markersize=3, linestyle='',
                     label="gpyfft[clFFT]")
        if "skcuda[cuFFT]" in results:
            y = results["skcuda[cuFFT]"]
            plt.plot(np.take(x, idx7a), np.take(y, idx7a), color='#0073FF', marker='o', markersize=3,
                     linestyle='', label="skcuda[cuFFT] (radix-7)")
            plt.plot(np.take(x, idx7b), np.take(y, idx7b), color='#0073FF', marker='+', markersize=3,
                     linestyle='', label="skcuda[cuFFT] (Bluestein/?)")
        if "vkFFT.opencl" in results:
            y = results["vkFFT.opencl"]
            plt.plot(np.take(x, idx13a), np.take(y, idx13a), color='#FF00F2', marker='o', markersize=3,
                     linestyle='', label="vkFFT.opencl (radix-13)")
            plt.plot(np.take(x, idx13b), np.take(y, idx13b), color='#FF00F2', marker='+', markersize=3,
                     linestyle='', label="vkFFT.opencl (Bluestein)")
        if "vkFFT.cuda" in results:
            y = results["vkFFT.cuda"]
            plt.plot(np.take(x, idx13a), np.take(y, idx13a), color='#FF8C00', marker='o', markersize=3,
                     linestyle='', label="vkFFT.cuda (radix-13)")
            plt.plot(np.take(x, idx13b), np.take(y, idx13b), color='#FF8C00', marker='+', markersize=3,
                     linestyle='', label="vkFFT.cuda (Bluestein)")
    else:
        x = results['n']
        if "gpyfft[clFFT]" in results:
            y = results["gpyfft[clFFT]"]
            plt.plot(x, y, color='#00FF0D', marker='o', markersize=3, linestyle='', label="gpyfft[clFFT]")
        if "skcuda[cuFFT]" in results:
            y = results["skcuda[cuFFT]"]
            plt.plot(x, y, color='#0073FF', marker='o', markersize=3, linestyle='', label="skcuda[cuFFT]")
        if "vkFFT.opencl" in results:
            y = results["vkFFT.opencl"]
            plt.plot(x, y, color='#FF00F2', marker='o', markersize=3, linestyle='', label="vkFFT.opencl")
        if "vkFFT.cuda" in results:
            y = results["vkFFT.cuda"]
            plt.plot(x, y, color='#FF8C00', marker='o', markersize=3, linestyle='', label="vkFFT.cuda")

    plt.legend(loc=legend_loc, fontsize=10)
    plt.xlabel("FFT size", fontsize=12)
    plt.ylabel("idealised throughput [Gbytes/s]", fontsize=12)
    if "skcuda[cuFFT]" in results:
        has_skcuda, cufft_version = test_skcuda()
        cufft_version = ", cuFFT " + cufft_version
    else:
        cufft_version = ""
    if "vkFFT.cuda" in results:
        has_pyvkfft_cuda, cu_version_compile, cu_version_driver, cu_version_runtime = test_pyvkfft_cuda()
        cu_version = ", CUDA driver %s runtime %s" % (cu_version_driver, cu_version_runtime)
    else:
        cu_version = ""
    vkfft_git_v = '' if 'unknown' in vkfft_git_version() else f'[{vkfft_git_version()}]'
    plt.suptitle("%dD FFT speed [%s, VkFFT %s%s%s%s]" % (ndim, gpu_name_real, vkfft_version(),
                                                         vkfft_git_v, cu_version, cufft_version),
                 fontsize=12)
    plt.title("Batched FFTs, 'Ideal' throughput assumes one r+w operation per FFT axis [%s, %s]" %
              (platform.platform(), platform.node()), fontsize=10)
    plt.grid(which='both', alpha=0.3)
    plt.xlim(0)
    plt.ylim(0)
    plt.tight_layout()

    # Force refresh
    plt.draw()
    plt.gcf().canvas.draw()
    plt.pause(.001)


def init_results(has_pyvkfft_opencl, has_pyvkfft_cuda, has_skcuda, has_gpyfft):
    results = {"n": [], "maxprime": []}
    if "vkFFT.opencl" not in results and has_pyvkfft_opencl:
        results["vkFFT.opencl"] = []
        results["vkFFT.opencl-dt"] = []
    if "gpyfft[clFFT]" not in results and has_gpyfft:
        results["gpyfft[clFFT]"] = []
        results["gpyfft[clFFT]-dt"] = []
    if "vkFFT.cuda" not in results and has_pyvkfft_cuda:
        results["vkFFT.cuda"] = []
        results["vkFFT.cuda-dt"] = []
    if "skcuda[cuFFT]" not in results and has_skcuda:
        results["skcuda[cuFFT]"] = []
        results["skcuda[cuFFT]-dt"] = []
    return results


def run(nmin, nmax, radix_max, ndim, precision="single", nb_repeat=3, nb_loop=1, gpu_name=None,
        batch=True, opencl_platform=None, figsize=(16, 8),
        has_pyvkfft_opencl=None, has_pyvkfft_cuda=None, has_gpyfft=None, has_skcuda=None,
        r2c=False, dct=False, dst=False, inplace=True):
    """
    Run the benchmark, measuring the idealised memory throughput (assuming a single
    read+write operation per axis) for an inplace C2C transform using different
    fft backends available.
    Note that each test is made in a separate individual process, so this can
    take a long time.

    :param nmin: smallest size N of the array, e.g. with a shape (batch, N, N)
        for a 2D transform.
    :param nmax: largest size N for the array.
    :param radix_max: maximum radix for the tested sizes. Use a large value (1e7)
        to test all sizes regardless of the prime decomposition.
    :param precision: either 'single' or 'double'
    :param nb_repeat: number of times each fft+ifft cycle is performed, the best
        timing is kept
    :param gpu_name: name or substring (case-insensitive) of the GPU to use.
        If None, the first found will be used.
    :param batch: if True (the default), all transforms are batched so that
        the array size is large enough to yield a measurable transform time. Each
        array takes a shape e.g. (batch, N, N) for a 2D transform.
    :param opencl_platform: name or substring (case-insensitive) of the OpenCL
        platform to use. If None, the first found will be used.
    :param figsize: figure size for plotting. Set to None to disable plotting.
    :param has_pyvkfft_opencl: if True, will test pvkfft.opencl. If None,
        will be automatically detected
    :param has_pyvkfft_cuda: if True, will test pvkfft.cuda. If None,
        will be automatically detected
    :param has_gpyfft: if True, will test gpyfft (clFFT). If None,
        will be automatically detected
    :param has_skcuda: if True, will test scikit.cuda (cuFFT). If None,
        will be automatically detected
    :param r2c: if True, test an r2c transform
    :param dct: test DCT of type 1,2,3 or 4
    :param dst: test DST of type 1,2,3 or 4
    :param inplace: if True, test inplace transforms
    """
    if has_pyvkfft_opencl is None:
        has_pyvkfft_opencl = test_pyvkfft_opencl()
    if has_pyvkfft_cuda is None:
        has_pyvkfft_cuda, cu_version_compile, cu_version_driver, cu_version_runtime = test_pyvkfft_cuda()
    if has_skcuda is None:
        has_skcuda, cufft_version = test_skcuda()
    if has_gpyfft is None:
        has_gpyfft = test_gpyfft()

    results = init_results(has_pyvkfft_opencl, has_pyvkfft_cuda, has_skcuda, has_gpyfft)
    if ndim == 1:
        header_results = "    1 x batch x     N [%dD]" % (ndim)
    elif ndim == 2:
        header_results = "batch x     N x     N [%dD]" % (ndim)
    else:
        header_results = "batch x    N x    N x    N [%dD]" % (ndim)
    for b in results.keys():
        if b not in ["n", "maxprime"] and "-dt" not in b:
            header_results += "%17s  " % b

    s = f"DCT{dct}" if dct else f"DST{dst}" if dst else "R2C" if r2c else "C2C"
    if inplace:
        s = "inplace " + s

    print(f"Gbytes/s and time given for a couple (FFT, iFFT) of {s}, "
          f"dtype={np.dtype(np.complex64).name}")
    print()
    print(header_results)

    if figsize is not None:
        fig = plt.figure(figsize=figsize)

    dtype = np.complex128 if precision == 'double' else np.complex64
    gpu_name_real_ok = None
    gpu_name_real = None

    for n in range(nmin, nmax + 1):
        maxprime = max(primes(n))
        if maxprime > radix_max:
            continue
        results["n"].append(n)
        results["maxprime"].append(maxprime)
        if batch:
            # Estimate batch size to last 0.05s with at least 100 GB/s
            nb = int(round(0.05 * 100 / (n ** ndim * np.dtype(dtype).itemsize * ndim * 2 * 2 / 1024 ** 3)))
            nb = max(nb, 1)
            nb = min(nb, 99999)
        else:
            nb = 1

        if ndim == 1:
            sh = 1, nb, n
        elif ndim == 2:
            sh = nb, n, n
        else:
            sh = nb, n, n, n

        vkfft_str = None
        # OpenCL backends
        if has_pyvkfft_opencl:
            res = bench_pyvkfft_opencl(sh, precision, ndim, nb_repeat, nb_loop, gpu_name, opencl_platform,
                                       r2c=r2c, dct=dct, dst=dst, inplace=inplace)
            dt = res['dt']
            gbps = res['gbps']
            gpu_name_real = res['gpu_name_real']
            platform_name_real = res['platform_name_real']
            vkfft_str = res['vkfft_str']
            results["vkFFT.opencl"].append(gbps)
            results["vkFFT.opencl-dt"].append(dt)

        if gpu_name_real_ok is None and gpu_name_real is not None:
            gpu_name_real_ok = gpu_name_real

        if has_gpyfft:
            res = bench_gpyfft(sh, precision, ndim, nb_repeat, nb_loop, gpu_name, opencl_platform)
            dt = res['dt']
            gbps = res['gbps']
            gpu_name_real = res['gpu_name_real']
            platform_name_real = res['platform_name_real']
            results["gpyfft[clFFT]"].append(gbps)
            results["gpyfft[clFFT]-dt"].append(dt)

        # CUDA backends
        if has_pyvkfft_cuda:
            res = bench_pyvkfft_cuda(sh, precision, ndim, nb_repeat, nb_loop, gpu_name)
            dt = res['dt']
            gbps = res['gbps']
            gpu_name_real = res['gpu_name_real']
            vkfft_str = res['vkfft_str']
            results["vkFFT.cuda"].append(gbps)
            results["vkFFT.cuda-dt"].append(dt)

        if gpu_name_real_ok is None and gpu_name_real is not None:
            gpu_name_real_ok = gpu_name_real

        if has_skcuda:
            res = bench_skcuda(sh, precision, ndim, nb_repeat, nb_loop, gpu_name)
            dt = res['dt']
            gbps = res['gbps']
            gpu_name_real = res['gpu_name_real']
            results["skcuda[cuFFT]"].append(gbps)
            results["skcuda[cuFFT]-dt"].append(dt)

        # text output
        if ndim == 3:
            r = " %4d x %4d x %4d x %4d      " % sh
        else:
            r = "%5d x %5d x %5d      " % sh
        for b in results.keys():
            if b not in ["n", "maxprime"] and "-dt" not in b:
                dt = results[b + '-dt'][-1] / nb
                if dt < 1e-3:
                    r += "%7.2f [%6.2f µs]" % (results[b][-1], dt * 1e6)
                elif dt > 1:
                    r += "%7.2f [%6.2f  s]" % (results[b][-1], dt)
                else:
                    r += "%7.2f [%6.2f ms]" % (results[b][-1], dt * 1000)
        if vkfft_str is not None:
            r += f" [VkFFT: {vkfft_str}]"
        print(r)

        if len(results['n']) % 10 == 9 and figsize is not None:
            plot_benchmark(results, ndim, gpu_name_real_ok, radix_max, legend_loc="upper right",
                           fig=fig)

    if figsize is not None:
        plot_benchmark(results, ndim, gpu_name_real_ok, radix_max, "upper right", fig=fig)
        r = "-radix%d" % radix_max if radix_max < 100 else ""
        figname = 'benchmark-%dDFFT%s-%s-%s-%s-%s.png' % (ndim, gpu_name_real_ok.replace(' ', '_'), r,
                                                          platform.platform(), platform.node(),
                                                          strftime("%Y-%m-%d-%Hh%M", localtime()))
        plt.savefig(figname)
        print("Saved benchmark figure to: \n    %s" % figname)
    print()
    return results


if __name__ == '__main__':
    # res = run(nmin=32, nmax=256, radix_max=7, ndim=2, gpu_name=None)
    # res = run(nmin=32, nmax=256, radix_max=7, ndim=2, gpu_name=None, batch=False)
    # res = run(nmin=128, nmax=256, radix_max=3, ndim=2, gpu_name=None, figsize=None)
    # res = run(nmin=128, nmax=256, radix_max=3, ndim=2, gpu_name=None, figsize=None, r2c=True)
    # res = run(nmin=128, nmax=256, radix_max=3, ndim=2, gpu_name=None, figsize=None, r2c=True, inplace=False)
    res = run(nmin=128, nmax=256, radix_max=3, ndim=2, gpu_name=None, figsize=None, has_gpyfft=False,
              has_skcuda=False)
    res = run(nmin=128, nmax=256, radix_max=3, ndim=2, gpu_name=None, figsize=None, has_gpyfft=False,
              has_skcuda=False, r2c=True)
    res = run(nmin=128, nmax=256, radix_max=3, ndim=2, gpu_name=None, figsize=None, has_gpyfft=False,
              has_skcuda=False, r2c=True, inplace=False)
