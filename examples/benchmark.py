import numpy as np
from sys import argv
##### Benchmark configuration 

# all parameters below can be replaced from the command-line

# If None, first GPU will be selected. Otherwise, give part of the GPU name string e.g. "V100"
gpu_name = None
ndim = 2  # Dimensions for the FFT (1, 2 or 3)
nmax = 2048  # Maximum FFT size (e.g. 512 for 3D, 4096 for 2D,...) - nmax is included
dtype = np.complex64  # Data type

# Largest allowed prime factor: use 2 or 3 for quick tests
# CuFFT supports radix-7 and VkFFT radix-13, and both allow larger primes using Bluestein algorithm
# [in fact CuFFT also supports an undocumented number or radix transforms with prime numbers <=127]
# clFFT supports radix-13.
# Use a number >= nmax to test non-radix transforms (these will be skipped for gpyfft)
radix_max = 3  # 1e7, 7, 13

cl_platform = None  # If None, the first platform with a GPU is selected. Otherwise match part of the platform name

# Read parameters from the command-line
for arg in argv:
    if 'gpu=' in arg:
        gpu_name = arg.split('gpu=')[-1]
    if 'gpu_name=' in arg:
        gpu_name = arg.split('gpu_name=')[-1]
    if 'ndim=' in arg:
        ndim = int(arg.split('ndim=')[-1])
    if 'nmax=' in arg:
        nmax = int(arg.split('nmax=')[-1])
    if 'dtype=' in arg:
        dtype = eval(arg.split('dtype=')[-1])
    if 'radix_max=' in arg:
        radix_max = int(arg.split('radix_max=')[-1])
    if 'cl_platform=' in arg:
        cl_platform = arg.split('cl_platform=')[-1]

print("GPU:", gpu_name)
print("ndim:", ndim)
print("nmax:", nmax)
print("dtype:", dtype)
print("radix_max:", radix_max)
print("cl_platform:", cl_platform)

##### Secondary parameters
nb_repeat = 3  # Perform nb_repeat tests, keep best time

import os
import platform
import gc
from itertools import permutations

try:
    import pycuda.driver as cu_drv
    import pycuda.gpuarray as cua
    from pycuda import curandom
    import pyvkfft.cuda
    from pyvkfft.cuda import primes, VkFFTApp as cuVkFFTApp

    has_pyvkfft_cuda = True
except ImportError:
    has_pyvkfft_cuda = False

try:
    import pyopencl as cl
    import pyopencl.array as cla
    from pyopencl import clrandom
    import pyvkfft.opencl
    from pyvkfft.opencl import primes, VkFFTApp as clVkFFTApp

    has_pyvkfft_opencl = True
except ImportError:
    has_pyvkfft_opencl = False

try:
    import pycuda.autoinit
    import pycuda.driver as cu_drv
    import pycuda.gpuarray as cua
    from pycuda import curandom
    import skcuda.fft as cu_fft

    has_skcuda = True
except:
    has_skcuda = False

try:
    import pyopencl as cl
    import pyopencl.array as cla
    from pyopencl import clrandom
    import gpyfft

    has_gpyfft = True
except:
    has_gpyfft = False

import matplotlib.pyplot as plt
import timeit
from time import localtime, strftime

# Use the following only to deactivate some backends for faster testing
# has_pyvkfft_opencl = False
# has_pyvkfft_cuda = False
# has_skcuda = False
# has_gpyfft = False          # gpyfft/clFFT is significantly slower

gpu_name_real = None
if has_pyvkfft_opencl or has_gpyfft:
    # Create some context on the first available GPU
    if 'PYOPENCL_CTX' in os.environ:
        cl_ctx = cl.create_some_context()
    else:
        cl_ctx = None
        # Find the first OpenCL GPU available and use it, unless
        for p in cl.get_platforms():
            if cl_platform is not None:
                if cl_platform.lower() not in p.name.lower():
                    continue
            for d in p.get_devices():
                if d.type & cl.device_type.GPU == 0:
                    continue
                if gpu_name is not None:
                    if gpu_name.lower() not in d.name.lower():
                        continue
                gpu_name_real = d.name
                print("Selected OpenCL device: %s [%s]" % (d.name, p.name))
                cl_ctx = cl.Context(devices=(d,))
                cl_platform_real = p.name
                break
            if cl_ctx is not None:
                break
    cq = cl.CommandQueue(cl_ctx)

if has_pyvkfft_cuda or has_skcuda:
    if gpu_name is None:
        d = cu_drv.Device(0)
        gpu_name_real = d.name()
        print("Selected  CUDA  device: ", d.name())
        cu_ctx = d.make_context()
    else:
        for i in range(cu_drv.Device.count()):
            d = cu_drv.Device(i)
            if gpu_name.lower() in d.name().lower():
                gpu_name_real = d.name()
                print("Selected  CUDA  device: ", d.name())
                cu_ctx = d.make_context()
                break

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

if ndim == 1:
    header_results = "    1 x batch x     N [%dD]" % (ndim)
elif ndim == 2:
    header_results = "batch x     N x     N [%dD]" % (ndim)
else:
    header_results = "batch x    N x    N x    N [%dD]" % (ndim)
for b in results.keys():
    if b not in ["n", "maxprime"] and "-dt" not in b:
        header_results += "%17s  " % b

print("Gbytes/s and time given for a couple (FFT, iFFT), dtype=%s" % np.dtype(np.complex64).name)
print()
print(header_results)

for n in range(32, nmax + 1):
    maxprime = max(primes(n))
    if maxprime > radix_max:
        continue
    results["n"].append(n)
    results["maxprime"].append(maxprime)
    # Estimate batch size to last 0.05s with at least 100 GB/s
    nb = int(round(0.05 * 100 / (n ** ndim * np.dtype(dtype).itemsize * ndim * 2 * 2 / 1024 ** 3)))
    nb = max(nb, 1)
    nb = min(nb, 99999)
    # print("%4d (nb=%4d)"%(n, nb))

    if ndim == 1:
        sh = 1, nb, n
    elif ndim == 2:
        sh = nb, n, n
    else:
        sh = nb, n, n, n

    # OpenCL backends
    if has_pyvkfft_opencl or has_gpyfft:
        d = clrandom.rand(cq, shape=sh, dtype=np.float32).astype(dtype)

    if has_pyvkfft_opencl:
        dt = 0
        try:
            app = clVkFFTApp(d.shape, d.dtype, queue=cq, ndim=ndim)
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
            del app
            gbps = d.nbytes * ndim * 2 * 2 / dt / 1024 ** 3
        except:
            gbps = 0
        results["vkFFT.opencl"].append(gbps)
        results["vkFFT.opencl-dt"].append(dt)
        gc.collect()

    if has_gpyfft:
        if maxprime > 13:
            results["gpyfft[clFFT]"].append(0)
            results["gpyfft[clFFT]-dt"].append(0)
        else:
            dt = 0
            for axes in permutations([-1, -2, -3][:ndim]):
                gpyfft_plan = gpyfft.FFT(cl_ctx, cq, d, None, axes=axes)
                # Shuffle axes order to find fastest transform
                for i in range(nb_repeat):
                    cq.finish()
                    t0 = timeit.default_timer()
                    gpyfft_plan.enqueue(forward=True)
                    gpyfft_plan.enqueue(forward=False)
                    cq.finish()
                    dt1 = timeit.default_timer() - t0
                    if dt == 0:
                        dt = dt1
                    elif dt1 < dt:
                        dt = dt1
                del gpyfft_plan
            gbps = d.nbytes * ndim * 2 * 2 / dt / 1024 ** 3
            # print("%4d %4dx%4d 2D FFT+iFFT dt=%6.2f ms %7.2f Gbytes/s [gpyfft[clFFT]]  [nb=%4d]" %
            #      (nz, n, n, dt / nb * 1000, gbps, nb))
            results["gpyfft[clFFT]"].append(gbps)
            results["gpyfft[clFFT]-dt"].append(dt)

    if has_pyvkfft_opencl or has_gpyfft:
        d.data.release()
        del d
        gc.collect()

    # CUDA backends
    if has_pyvkfft_cuda or has_pyvkfft_cuda:
        # d = curandom.rand(shape=sh, dtype=np.float32).astype(dtype)
        d = cua.zeros(sh, dtype=dtype)

    if has_pyvkfft_cuda:
        if True:
            app = cuVkFFTApp(d.shape, d.dtype, ndim=ndim)
            dt = 0
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
        # except:
        #    gbps = 0
        results["vkFFT.cuda"].append(gbps)
        results["vkFFT.cuda-dt"].append(dt)
        gc.collect()
        gc.collect()

    if has_skcuda:
        if ndim == 1:
            plan = cu_fft.Plan(n, dtype, dtype, batch=nb)
        elif ndim == 2:
            plan = cu_fft.Plan((n, n), dtype, dtype, batch=nb)
        else:
            plan = cu_fft.Plan((n, n, n), dtype, dtype, batch=nb)
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
        # print("%4d %4dx%4d 2D FFT+iFFT dt=%6.2f ms %7.2f Gbytes/s [skcuda[cuFFT]]    [nb=%4d]" %
        #      (nz, n, n, dt / nb * 1000, gbps, nb))
        del plan
        results["skcuda[cuFFT]"].append(gbps)
        results["skcuda[cuFFT]-dt"].append(dt)

    if has_pyvkfft_cuda or has_pyvkfft_cuda:
        d.gpudata.free()
        del d
        gc.collect()
        gc.collect()

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
    print(r)

plt.figure(figsize=(15, 8))
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

plt.legend(loc="lower left", fontsize=10)  # "top right"
plt.xlabel("FFT size", fontsize=12)
plt.ylabel("idealised throughput [Gbytes/s]", fontsize=12)
plt.suptitle("%dD FFT speed [%s, %s, %s]" % (ndim, gpu_name_real, platform.platform(),
                                             platform.node()), fontsize=12)
plt.title("Batched FFTs, 'Ideal' throughput assumes one r+w operation per FFT axis", fontsize=10)
plt.grid(which='both', alpha=0.3)
plt.xlim(0)
plt.ylim(0)
plt.tight_layout()

file_name = 'benchmark-%dDFFT-%s-%s-%s-%s.png' % (ndim, gpu_name_real.replace(' ', '_'),
                                                  platform.platform(), platform.node(),
                                                  strftime("%Y-%m-%d-%Hh%M", localtime()))
plt.savefig(file_name)
print("Saved plot to %s" % file_name)

if has_pyvkfft_cuda or has_skcuda:
    cu_ctx.pop()
