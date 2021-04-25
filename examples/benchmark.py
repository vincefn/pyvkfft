import numpy as np

##### Benchmark configuration 
gpu_name = None       # If None, first GPU will be selected. Otherwise, give part of the GPU name string
ndim = 2              # Dimensions for the FFT (1, 2 or 3)
nmax = 4096           # Maximum FFT size (e.g. 512 for 3D, 4096 for 2D,...) - nmax is included
dtype = np.complex64  # Data type
radix_max = 7         # Largest allowed prime factor: use 2 for quick tests or 7 (13 is also possible)

cl_platform = "nvidia"# If None, the first platform with a GPU is selected. Otherwise match part of the platform name

##### Secondary parameters
nb_repeat = 3         # Perform nb_repeat tests, keep best time

# number of parallel arrays for 2D (nz, n, n) and 1D (nz, nz, n) transforms
nz = 16               


import os
import platform
import gc
from itertools import permutations

try:
    import pycuda.driver as cu_drv
    import pycuda.gpuarray as cua
    from pycuda import curandom
    import pyvkfft.cuda
    from pyvkfft.cuda import primes, VkFFTApp as  cuVkFFTApp
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



results = {"n": []}
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
    header_results = "%4d x%4d x%4s [%dD]" % (nz, nz, "N", ndim)
elif ndim ==2:
    header_results = "%4d x%4s x%4s [%dD]" % (nz, "N", "N", ndim)
else:
    header_results = "%4s x%4s x%4s [%dD]" % ("N", "N", "N", ndim)
for b in results.keys():
    if b != "n" and "-dt" not in b:
        header_results += "%17s  " % b


print("Gbytes/s and time given for a couple (FFT, iFFT), dtype=%s" % np.dtype(np.complex64).name)
print()
print(header_results)


# Only test up to prime factors equal to 7 (cuFFT)
for n in range(16, nmax+1):
    if max(primes(n)) > radix_max:
        continue
    results["n"].append(n)
    # Estimate number of repeats to last 0.1s with at least 100 GB/s
    nb = int(round(0.1 * 100 / (nz**(3-ndim) * n ** ndim * np.dtype(dtype).itemsize * ndim * 2 * 2 / 1024 ** 3)))
    nb = max(nb, 1)
    nb = min(nb, 1000)
    # print("%4d (nb=%4d)"%(n, nb))
    
    if ndim == 1:
        sh = nz, nz, n
    elif ndim == 2:
        sh = nz, n, n
    else:
        sh = n, n, n
    
    # OpenCL backends
    if has_pyvkfft_opencl or has_gpyfft:
        d = clrandom.rand(cq, shape=sh, dtype=np.float32).astype(dtype)
    
    if has_pyvkfft_opencl:
        dt = 0
        try:
            app= clVkFFTApp(d.shape, d.dtype, queue=cq, ndim=ndim)
            for i in range(nb_repeat):
                cq.finish()
                t0 = timeit.default_timer()
                for i in range(nb):
                    d = app.ifft(d)
                    d = app.fft(d)
                cq.finish()
                dt1 = timeit.default_timer() - t0
                if dt == 0:
                    dt = dt1
                elif dt1< dt:
                    dt = dt1
            #print("%4d %4dx%4d 2D FFT+iFFT dt=%6.2f ms %7.2f Gbytes/s [pyvkfft.opencl]  [nb=%4d]" %
            #      (nz, n, n, dt / nb * 1000, gbps, nb))
            del app
            gbps = d.nbytes * nb * ndim * 2 * 2 / dt / 1024 ** 3
        except:
            gbps = 0
        results["vkFFT.opencl"].append(gbps)
        results["vkFFT.opencl-dt"].append(dt)
        gc.collect()
    
    if has_gpyfft:
        dt = 0
        for axes in permutations([-1, -2, -3][:ndim]):
            gpyfft_plan = gpyfft.FFT(cl_ctx, cq, d, None, axes=axes)
            # Shuffle axes order to find fastest transform
            for i in range(nb_repeat):
                cq.finish()
                t0 = timeit.default_timer()
                for i in range(nb):
                    gpyfft_plan.enqueue(forward=True)
                    gpyfft_plan.enqueue(forward=False)
                cq.finish()
                dt1 = timeit.default_timer() - t0
                if dt == 0:
                    dt = dt1
                elif dt1< dt:
                    dt = dt1
            del gpyfft_plan
        gbps = d.nbytes * nb * ndim * 2 * 2 / dt / 1024 ** 3
        #print("%4d %4dx%4d 2D FFT+iFFT dt=%6.2f ms %7.2f Gbytes/s [gpyfft[clFFT]]  [nb=%4d]" %
        #      (nz, n, n, dt / nb * 1000, gbps, nb))
        results["gpyfft[clFFT]"].append(gbps)
        results["gpyfft[clFFT]-dt"].append(dt)

    if has_pyvkfft_opencl or has_gpyfft:
        d.data.release()
        del d
        gc.collect()
    
    # CUDA backends
    if has_pyvkfft_cuda or has_pyvkfft_cuda:
        d = curandom.rand(shape=sh, dtype=np.float32).astype(dtype)

    if has_pyvkfft_cuda:
        dt = 0
        try:
            app= cuVkFFTApp(d.shape, d.dtype, ndim=ndim)
            for i in range(nb_repeat):
                cu_ctx.synchronize()
                t0 = timeit.default_timer()
                for i in range(nb):
                    d = app.ifft(d)
                    d = app.fft(d)
                cu_ctx.synchronize()
                dt1 = timeit.default_timer() - t0
                if dt == 0:
                    dt = dt1
                elif dt1< dt:
                    dt = dt1
            #print("%4d %4dx%4d 2D FFT+iFFT dt=%6.2f ms %7.2f Gbytes/s [pyvkfft.cuda]    [nb=%4d]" %
            #      (nz, n, n, dt / nb * 1000, gbps, nb))
            del app
            gbps = d.nbytes * nb * ndim * 2 * 2 / dt / 1024 ** 3
        except:
            gbps = 0
        results["vkFFT.cuda"].append(gbps)
        results["vkFFT.cuda-dt"].append(dt)
        gc.collect()

    if has_skcuda:
        if ndim == 1:
            plan = cu_fft.Plan(n, dtype, dtype, batch=nz*nz)
        elif ndim == 2:
            plan = cu_fft.Plan((n,n), dtype, dtype, batch=nz)
        else:
            plan = cu_fft.Plan((n,n,n), dtype, dtype, batch=1)
        dt = 0
        for i in range(nb_repeat):
            cu_ctx.synchronize()
            t0 = timeit.default_timer()
            for i in range(nb):
                cu_fft.fft(d, d, plan)
                cu_fft.ifft(d, d, plan)
            cu_ctx.synchronize()
            dt1 = timeit.default_timer() - t0
            if dt == 0:
                dt = dt1
            elif dt1< dt:
                dt = dt1
        gbps = d.nbytes * nb * ndim * 2 * 2 / dt / 1024 ** 3
        #print("%4d %4dx%4d 2D FFT+iFFT dt=%6.2f ms %7.2f Gbytes/s [skcuda[cuFFT]]    [nb=%4d]" %
        #      (nz, n, n, dt / nb * 1000, gbps, nb))
        del plan
        results["skcuda[cuFFT]"].append(gbps)
        results["skcuda[cuFFT]-dt"].append(dt)

        
    if has_pyvkfft_cuda or has_pyvkfft_cuda:
        d.gpudata.free()
        del d
        gc.collect()
    
    # text output
    r = "%4d x%4d x %4d      " % sh
    for b in results.keys():
        if b != "n" and "-dt" not in b:
            dt = results[b+'-dt'][-1] / nb
            if dt < 1e-3 :
                r += "%7.2f [%6.2f µs]" % (results[b][-1], dt * 1e6)
            elif dt > 1:
                r += "%7.2f [%6.2f  s]" % (results[b][-1], dt)
            else:
                r += "%7.2f [%6.2f ms]" % (results[b][-1], dt * 1000)
    print(r + "  [nb=%4d]"%nb)

plt.figure(figsize=(9.5, 8))
plt.clf()
x = results['n']
if "gpyfft[clFFT]" in results:
    y = results["gpyfft[clFFT]"]
    plt.plot(x, y, color='#00A000', marker='v', markersize=3, linestyle='', label="gpyfft[clFFT]")
if "skcuda[cuFFT]" in results:
    y = results["skcuda[cuFFT]"]
    plt.plot(x, y, color='#A00000', marker='^', markersize=3, linestyle='', label="skcuda[cuFFT]")
if "vkFFT.opencl" in results:
    y = results["vkFFT.opencl"]
    plt.plot(x, y, color='#00FF00', marker='o', markersize=3, linestyle='', label="vkFFT.opencl")
if "vkFFT.cuda" in results:
    y = results["vkFFT.cuda"]
    plt.plot(x, y, color='#FF0000', marker='o', markersize=3, linestyle='', label="vkFFT.cuda")

plt.legend(loc='lower right', fontsize=10)
plt.xlabel("FFT size", fontsize=12)
plt.ylabel("idealised throughput [Gbytes/s]", fontsize=12)
plt.suptitle("%dD FFT speed [%s, %s, %s]" % (ndim, gpu_name_real, platform.platform(),
                                             platform.node()), fontsize=12)
plt.title("'Ideal' throughput assumes one r+w operation per FFT axis", fontsize=10)
plt.grid(which='both', alpha=0.3)
plt.xlim(0)
plt.ylim(0)    
plt.tight_layout()
    

plt.savefig('benchmark-%dDFFT-%s-%s-%s.png'%(ndim, gpu_name_real.replace(' ','_'), 
                                             platform.platform(), platform.node()))

if has_pyvkfft_cuda or has_skcuda:
    cu_ctx.pop()
