# -*- coding: utf-8 -*-

# PyVkFFT
#   (c) 2021- : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr
#
#
# pyvkfft unit tests.
import sys
import itertools
import platform
import unittest
import multiprocessing
import threading
import sqlite3
import socket
import time
import timeit
from itertools import permutations
from copy import copy
import numpy as np

try:
    from scipy.datasets import ascent
except ImportError:
    try:
        from scipy.misc import ascent
    except ImportError:
        def ascent():
            return np.random.randint(0, 255, (512, 512))

from pyvkfft.version import __version__, vkfft_version, vkfft_git_version
from pyvkfft.base import primes, radix_gen_n
from pyvkfft.fft import fftn as vkfftn, ifftn as vkifftn, rfftn as vkrfftn, \
    irfftn as vkirfftn, dctn as vkdctn, idctn as vkidctn, \
    dstn as vkdstn, idstn as vkidstn, clear_vkfftapp_cache
from pyvkfft.accuracy import test_accuracy, test_accuracy_kwargs, fftn, init_ctx, gpu_ctx_dic, has_dct_ref, has_scipy
import pyvkfft.config


def find_gpu(backend):
    """
    Find available GPUs for a given backend.
    :param backend: either 'pycuda', 'pyopencl' or 'cupy'
    :return: a list of GPU devices
    """
    v = []
    if backend == "pycuda":
        if not has_pycuda:
            raise RuntimeError("find_gpu: backend=%s is not available" % backend)
        cu_drv.init()
        for i in range(cu_drv.Device.count()):
            v.append(cu_drv.Device(i))
    elif backend == "pyopencl":
        for p in cl.get_platforms():
            has_pocl = False
            if 'portable' in p.name.lower():
                # For now, skip POCL - has issues with a number of calculations
                has_pocl = True
                continue
            for d0 in p.get_devices():
                if d0.type & cl.device_type.GPU:
                    v.append(d0)
            if len(v) == 0 and has_pocl:
                # last resort - add pocl devices
                for p in cl.get_platforms():
                    if 'portable' in p.name.lower():
                        for d0 in p.get_devices():
                            if d0.type & cl.device_type.GPU:
                                v.append(d0)
    elif backend == "cupy":
        if not has_cupy:
            raise RuntimeError("find_gpu: backend=%s is not available" % backend)
        for i in range(cp.cuda.runtime.getDeviceCount()):
            v.append(cp.cuda.Device(i))
    else:
        raise RuntimeError("find_gpu: unknown backend ", backend)
    return v


try:
    import pycuda.gpuarray as cua

    import pycuda.driver as cu_drv
    from pyvkfft.cuda import VkFFTApp as cuVkFFTApp

    has_pycuda = True
    v_gpu_pycuda = find_gpu("pycuda")
except ImportError:
    has_pycuda = False
    v_gpu_pycuda = []

try:
    import cupy as cp

    has_cupy = True
    v_gpu_cupy = find_gpu("cupy")
except ImportError:
    has_cupy = False
    v_gpu_cupy = []

try:
    import pyopencl as cl
    import pyopencl.array as cla
    from pyvkfft.opencl import VkFFTApp as clVkFFTApp

    has_pyopencl = True
    try:
        v_gpu_pyopencl = find_gpu("pyopencl")
    except cl.Error:
        # Probably clGetPlatformIDs failed: PLATFORM_NOT_FOUND_KHR (no platform available)
        has_pyopencl = False
        v_gpu_pyopencl = []
except ImportError:
    has_pyopencl = False
    v_gpu_pyopencl = []


def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


class TestFFT(unittest.TestCase):
    gpu = None
    nproc = 1
    verbose = True
    colour = False
    opencl_platform = None
    vbackend = None

    def setUp(self) -> None:
        if self.vbackend is None:
            self.vbackend = []
            if has_pycuda:
                self.vbackend.append("pycuda")
            if has_cupy:
                self.vbackend.append("cupy")
            if has_pyopencl:
                self.vbackend.append("pyopencl")
        if "pyopencl" in self.vbackend and self.opencl_platform is None:
            # Try to exclude PoCL unless an opencl_platform was requested
            if len(v_gpu_pyopencl):
                if self.gpu is None:
                    self.opencl_platform = v_gpu_pyopencl[0].platform.name
                else:
                    for d in v_gpu_pyopencl:
                        if self.gpu.lower() in d.name.lower():
                            self.opencl_platform = d.platform.name
                            break

    def test_backend(self):
        self.assertTrue(has_pycuda or has_pyopencl or has_cupy,
                        "Either pycuda, pyopencl or cupy must be available")

    @unittest.skipIf(not (has_pycuda or has_cupy or has_pyopencl), "No OpenCL/CUDA backend is available")
    def test_simple_fft(self):
        """Test the simple fft API"""
        for backend in self.vbackend:
            with self.subTest(backend=backend):
                init_ctx(backend, gpu_name=self.gpu, opencl_platform=self.opencl_platform, verbose=False)
                a = ascent()[:256, :256]  # Crop to 256 to avoid DCT issues
                if backend == "pycuda":
                    dc = cua.to_gpu(a.astype(np.complex64))
                    dr = cua.to_gpu(a.astype(np.float32))
                elif backend == "cupy":
                    cp.cuda.Device(0).use()
                    dc = cp.array(a.astype(np.complex64))
                    dr = cp.array(a.astype(np.float32))
                else:
                    cq = gpu_ctx_dic["pyopencl"][2]
                    dc = cla.to_device(cq, a.astype(np.complex64))
                    dr = cla.to_device(cq, a.astype(np.float32))
                # C2C, new destination array
                d = vkfftn(dc)
                d = vkifftn(d)
                # C2C in-place
                d = vkfftn(d, d)
                d = vkifftn(d, d)
                # C2C out-of-place
                d2 = d.copy()
                d2 = vkfftn(d, d2)
                d = vkifftn(d2, d)

                # R2C, new destination array
                d = vkrfftn(dr)
                self.assertTrue(d.dtype == np.complex64)
                d = vkirfftn(d)
                self.assertTrue(d.dtype == np.float32)

                # R2C, inplace
                d = vkrfftn(dr, dr)
                self.assertTrue(d.dtype == np.complex64)
                d = vkirfftn(d, d)
                self.assertTrue(d.dtype == np.float32)

                # DCT, new destination array
                d = vkdctn(dr)
                d = vkidctn(d)

                # DCT, out-of-place
                d2 = dr.copy()
                d2 = vkdctn(dr, d2)
                dr = vkidctn(d2, dr)

                # DCT, inplace
                d = vkdctn(dr, dr)
                d = vkidctn(d, d)

                # DST, new destination array
                d = vkdstn(dr)
                d = vkidstn(d)

                # DST, out-of-place
                d2 = dr.copy()
                d2 = vkdstn(dr, d2)
                dr = vkidstn(d2, dr)

                # DCT, inplace
                d = vkdstn(dr, dr)
                d = vkidstn(d, d)
                if backend == "pycuda":
                    gpu_ctx_dic["pycuda"][1].pop()

    def run_fft(self, vbackend, vn, dims_max=4, ndim_max=3, shuffle_axes=True,
                vtype=(np.complex64, np.complex128),
                vlut="auto", vinplace=(True, False), vnorm=(0, 1),
                vr2c=(False,), vdct=(False,), vdst=(False,), verbose=False, dry_run=False,
                secondary_long_axis_size=0):
        """
        Run a series of tests
        :param vbackend: list of backends to test among "pycuda", "cupy and "pyopencl"
        :param vn: list of transform sizes to test
        :param dims_max: max number of dimensions for the array (up to 4)
        :param ndim_max: max transform dimension
        :param shuffle_axes: if True, all possible axes combinations will be tried for
            the given shape of the array and the number of transform dimensions, e.g.
            for a 3D array and ndim=2 this would try (-1, -2), (-1, -3) and (-2,-3).
            This applies only to C2C transforms.
        :param vtype: list of array types among float32, float64, complex64, complex128
        :param vlut: if "auto" (the default), will test useLUT=None and True, except for
            double precision where LUT is always enabled. Can be a list of values among
            None (uses VkFFT default), 0/False and 1/True.
        :param vinplace: a list among True and False
        :param vnorm: a list among 0, 1, and (for C2C only) "ortho"
        :param vr2c: a list among True, False to perform r2c tests
        :param vdct: a list among False/0, 1, 2, 3, 4 to test various DCT
        :param vdst: a list among False/0, 1, 2, 3, 4 to test various DST
        :param verbose: True or False - prints two lines per test (FFT and iFFT result)
        :param dry_run: if True, only count the number of test to run
        :param secondary_long_axis_size: if >0, then when testing for ndim>1, the test
            will be done by separating the long axis along the transform
            dimensions, only one axis being long and the other transformed axes
            will use the given size.
            Example: for a size and ndim=3, if secondary_long_axis_size=2,
            instead of testing with a shape (n,n,n), the 2D test will
            be done with shapes (2,2,n),(2,n,2) and (n,2,2).
            This is useful to test larger ndim>1 transforms (requiring multi-upload)
            without using up too much memory.
        :return: the number of tests performed, and the list of kwargs (dry run)
        """
        ct = 0
        vkwargs = []
        # Aggregate r2c, dct, dst in a single list
        vrcs = []
        for r2c in vr2c:
            if r2c is False:
                for dct in vdct:
                    if dct is False:
                        for dst in vdst:
                            vrcs.append((False, False, dst))
                    else:
                        vrcs.append((False, dct, False))
            else:
                vrcs.append((r2c, False, False))
        if verbose:
            print("\n backend transform     shape         axes         ndim    FFTAlgo NbUp"
                  "   type      lut?    inplace?      norm  C/F  FFT: L2 error     Linf"
                  "       < max  (Linf/max) unchanged? iFFT: L2   Linf      < max"
                  "  (Linf/max) unchanged? buffer status")
        for backend in vbackend:
            # We assume the context was already initialised by the calling function
            # init_ctx(backend, gpu_name=self.gpu, opencl_platform=self.opencl_platform, verbose=False)
            cq = gpu_ctx_dic["pyopencl"][2] if backend == "pyopencl" else None
            for n, dims in itertools.product(vn, range(1, dims_max + 1)):
                for ndim0 in range(1, min(dims, ndim_max) + 1):
                    for r2c, dct, dst in vrcs:
                        # Setup use of either ndim or axes, also test skipping dimensions
                        ndim_axes = [(ndim0, None)]
                        if shuffle_axes:
                            for p in permutations([1] * ndim0 + [0] * (dims - ndim0)):
                                axes = (-dims + np.nonzero(p)[0]).tolist()
                                if (None, axes) not in ndim_axes:
                                    # Fast axis must be transformed for r2c
                                    if not r2c or -1 in axes:
                                        ndim_axes.append((None, axes))
                        for ndim, axes in ndim_axes:
                            for dtype in vtype:
                                if axes is None:
                                    axes_numpy = list(range(dims))[-ndim:]
                                else:
                                    axes_numpy = axes

                                # Array shape
                                sh0 = [n] * dims

                                # Use only a size of 2 for non-transform axes
                                for ii in range(len(sh0)):
                                    if ii not in axes_numpy and (-len(sh0) + ii) not in axes_numpy:
                                        sh0[ii] = 2
                                # List of shapes for secondary_long_axis_size
                                if secondary_long_axis_size and len(axes_numpy) > 1:
                                    vsh = []
                                    for ii in range(len(axes_numpy)):
                                        tmpsh = copy(sh0)
                                        for iii in range(len(axes_numpy)):
                                            if iii != ii:
                                                tmpsh[axes_numpy[iii]] = secondary_long_axis_size
                                        vsh.append(tmpsh)
                                else:
                                    vsh = [sh0]

                                if not dry_run:
                                    if dtype in (np.float32, np.float64):
                                        d0 = np.random.uniform(-0.5, 0.5, vsh[0]).astype(dtype)
                                    else:
                                        d0 = (np.random.uniform(-0.5, 0.5, vsh[0])
                                              + 1j * np.random.uniform(-0.5, 0.5, vsh[0])).astype(dtype)
                                if vlut == "auto":
                                    if dtype in (np.float64, np.complex128):
                                        # By default, LUT is enabled for complex128, no need to test twice
                                        tmp = [None]
                                    else:
                                        tmp = [None, True]
                                else:
                                    tmp = vlut
                                for sh, use_lut, inplace, norm in itertools.product(vsh, tmp, vinplace, vnorm):
                                    if not dry_run:
                                        d0 = d0.reshape(sh)  # needed for secondary_long_axis_size shape change
                                    vorder = ['C', 'F']
                                    if dims == 1 or dims > 3:
                                        vorder = ['C']
                                    if r2c:
                                        if ndim is not None:
                                            if dims != ndim:
                                                vorder = ['C']
                                        if axes is not None:
                                            # TODO : also test F-order when ndim<dims but fast axis
                                            #  is transformed
                                            if len(axes) != dims:
                                                vorder = ['C']
                                        if backend == "cupy" and inplace:
                                            # cupy does not support returning a view of a float32
                                            # array as complex64 if the *last* axis is not
                                            # contiguous
                                            vorder = ['C']
                                    for order in vorder:
                                        with self.subTest(backend=backend, shape=sh, dims=dims, ndim=ndim,
                                                          axes=axes, dtype=np.dtype(dtype), norm=norm,
                                                          use_lut=use_lut, inplace=inplace,
                                                          r2c=r2c, dct=dct, dst=dst, order=order):
                                            ct += 1
                                            if not dry_run:
                                                res = \
                                                    test_accuracy(backend, sh, ndim, axes, dtype,
                                                                  inplace,
                                                                  norm, use_lut, r2c=r2c, dct=dct,
                                                                  dst=dst,
                                                                  gpu_name=self.gpu,
                                                                  opencl_platform=self.opencl_platform,
                                                                  stream=None, queue=cq,
                                                                  return_array=False, init_array=d0,
                                                                  verbose=verbose, order=order)
                                                npr = primes(n)
                                                ni, n2 = res["ni"], res["n2"]
                                                nii, n2i = res["nii"], res["n2i"]
                                                tol = res["tol"]
                                                src1 = res["src_unchanged_fft"]
                                                src2 = res["src_unchanged_ifft"]
                                                self.assertTrue(ni < tol,
                                                                "Accuracy mismatch after FFT, "
                                                                "n2=%8e ni=%8e>%8e" %
                                                                (n2, ni, tol))
                                                self.assertTrue(nii < tol,
                                                                "Accuracy mismatch after iFFT, "
                                                                "n2=%8e ni=%8e>%8e" %
                                                                (n2, nii, tol))
                                                if not inplace:
                                                    self.assertTrue(src1,
                                                                    "The source array was modified "
                                                                    "during the FFT")
                                                    nmaxr2c1d = 3072 * (1 + int(
                                                        dtype in (np.float32, np.complex64)))
                                                    if not r2c or (ndim == 1 and max(npr) <= 13) \
                                                            and n < nmaxr2c1d:
                                                        self.assertTrue(src2,
                                                                        "The source array was modified "
                                                                        "during the iFFT")
                                            else:
                                                kwargs = {"backend": backend, "shape": sh,
                                                          "ndim": ndim, "axes": axes,
                                                          "dtype": dtype, "inplace": inplace,
                                                          "norm": norm, "use_lut": use_lut,
                                                          "r2c": r2c, "dct": dct,
                                                          "dst": dst,
                                                          "gpu_name": self.gpu,
                                                          "opencl_platform": self.opencl_platform,
                                                          "stream": None,
                                                          "verbose": False, "order": order,
                                                          "colour_output": self.colour}
                                                vkwargs.append(kwargs)

        return ct, vkwargs

    def run_fft_parallel(self, vkwargs):
        if self.verbose:
            print("\n backend transform     shape         axes         ndim    FFTAlgo NbUp"
                  "   type      lut?    inplace?      norm  C/F  FFT: L2 error     Linf"
                  "       < max  (Linf/max) unchanged? iFFT: L2   Linf      < max"
                  "  (Linf/max) unchanged? buffer status")

        # Need to use spawn to handle the GPU context
        with multiprocessing.get_context('spawn').Pool(self.nproc) as pool:
            for res in pool.imap(test_accuracy_kwargs, vkwargs):
                with self.subTest(backend=res['backend'], shape=res['shape'], ndim=res['ndim'],
                                  axes=res['axes'], dtype=np.dtype(res['dtype']),
                                  norm=res['norm'], use_lut=res['use_lut'],
                                  inplace=res['inplace'], r2c=res['r2c'], dct=res['dct'],
                                  dst=res['dst'], order=res['order']):
                    n = max(res['shape'])
                    npr = primes(n)
                    ni, n2 = res["ni"], res["n2"]
                    nii, n2i = res["nii"], res["n2i"]
                    tol = res["tol"]
                    src1 = res["src_unchanged_fft"]
                    src2 = res["src_unchanged_ifft"]
                    if self.verbose:
                        print(res['str'])
                    self.assertTrue(ni < tol, "Accuracy mismatch after FFT, n2=%8e ni=%8e>%8e" % (n2, ni, tol))
                    self.assertTrue(nii < tol, "Accuracy mismatch after iFFT, n2=%8e ni=%8e>%8e" % (n2, nii, tol))
                    if not res['inplace']:
                        self.assertTrue(src1, "The source array was modified during the FFT")
                        if platform.system() == 'Darwin':
                            nmaxr2c1d = 2048 * (1 + int(res['dtype'] in (np.float32, np.complex64)))
                        else:
                            nmaxr2c1d = 3072 * (1 + int(res['dtype'] in (np.float32, np.complex64)))
                        if not res['r2c'] or (res['ndim'] == 1 and max(npr) <= 13) and n < nmaxr2c1d:
                            # Only 1D radix C2R do not alter the source array,
                            # if n<= 3072 or 6144 (assuming 48kb shared memory)
                            self.assertTrue(src2, "The source array was modified during the iFFT")

    @staticmethod
    def backend_info(backend):
        s = ""
        if backend == "pycuda":
            d = gpu_ctx_dic[backend][0]
            s = f"{backend}, gpu: {d.name()}"
        elif backend == "pyopencl":
            d = gpu_ctx_dic[backend][0]
            s = f"{backend}, gpu: {d.name}, platform: {d.platform.name}"
        elif backend == "cupy":
            d = gpu_ctx_dic[backend]
            s = f"{backend}, gpu: {cp.cuda.runtime.getDeviceProperties(d.id)['name'].decode()}"
        return s

    @unittest.skipIf(not (has_pycuda or has_cupy or has_pyopencl), "No OpenCL/CUDA backend is available")
    def test_c2c(self):
        """Run C2C tests"""
        for backend in self.vbackend:
            init_ctx(backend, gpu_name=self.gpu, opencl_platform=self.opencl_platform, verbose=False)
            has_cl_fp64 = gpu_ctx_dic["pyopencl"][3] if backend == "pyopencl" else True
            ct = 0
            vkwargs = []
            for dry_run in [True, False]:
                vtype = (np.complex64, np.complex128)
                if backend == "pyopencl" and not has_cl_fp64:
                    vtype = (np.complex64,)
                v = self.verbose and not dry_run
                if dry_run or self.nproc == 1:
                    tmp = self.run_fft([backend], [15, 17], dims_max=5, ndim_max=5,
                                       vtype=vtype, verbose=v, dry_run=dry_run, shuffle_axes=True)
                    ct += tmp[0]
                    vkwargs += tmp[1]
                    # Avoid large array sizes for 30, 34
                    tmp = self.run_fft([backend], [30, 34], dims_max=5, ndim_max=5,
                                       vtype=vtype, verbose=v, dry_run=dry_run, shuffle_axes=True,
                                       secondary_long_axis_size=3)
                    ct += tmp[0]
                    vkwargs += tmp[1]
                    # Test larger sizes for 1D and 2D transforms
                    # 2988 decomposes in a Sophie Germain safe prime, hence its inclusion.
                    # Includes very long 1-2D transform - radix, Bluestein and Rader with 2 and 3 uploads
                    # Use secondary_long_axis_size to avoid too large overall sizes, but still try
                    # the transform up to ndim=2
                    tmp = self.run_fft([backend],
                                       [808, 2988, 4200, 8232, 8193, 8194,
                                        8232 * 128, 8193 * 128, 8194 * 128, 131072],
                                       vtype=vtype, dims_max=2, ndim_max=2, verbose=v,
                                       dry_run=dry_run, shuffle_axes=True, secondary_long_axis_size=3)
                    ct += tmp[0]
                    vkwargs += tmp[1]
                else:
                    self.run_fft_parallel(vkwargs)
                if dry_run and self.verbose:
                    print(f"Running {ct} C2C tests (backend: {self.backend_info(backend)})")

    @unittest.skipIf(not (has_pycuda or has_cupy or has_pyopencl), "No OpenCL/CUDA backend is available")
    def test_r2c(self):
        """Run R2C tests"""
        for backend in self.vbackend:
            init_ctx(backend, gpu_name=self.gpu, opencl_platform=self.opencl_platform, verbose=False)
            has_cl_fp64 = gpu_ctx_dic["pyopencl"][3] if backend == "pyopencl" else True
            ct = 0
            vkwargs = []
            for dry_run in [True, False]:
                vtype = (np.float32, np.float64)
                if backend == "pyopencl" and not has_cl_fp64:
                    vtype = (np.float32,)
                v = self.verbose and not dry_run
                if dry_run or self.nproc == 1:
                    tmp = self.run_fft([backend], [15, 17], dims_max=4, ndim_max=3,
                                       vtype=vtype, vr2c=(True,), verbose=v, dry_run=dry_run,
                                       shuffle_axes=True)
                    ct += tmp[0]
                    vkwargs += tmp[1]
                    # Avoid large array sizes for 30, 34
                    tmp = self.run_fft([backend], [30, 34], dims_max=4, ndim_max=4,
                                       vtype=vtype, vr2c=(True,), verbose=v, dry_run=dry_run,
                                       shuffle_axes=True, secondary_long_axis_size=3)
                    ct += tmp[0]
                    vkwargs += tmp[1]
                    # Test larger sizes for 1D and 2D transforms
                    # 2988 decomposes in a Sophie Germain safe prime, hence its inclusion.
                    # Includes very long 1-2D transform - radix, Bluestein and Rader with 2 and 3 uploads
                    # Use secondary_long_axis_size to avoid too large overall sizes, but still try
                    # the transform up to ndim=2
                    tmp = self.run_fft([backend],
                                       [808, 2988, 4200, 8232, 8193, 8194,
                                        8232 * 128, 8193 * 128, 8194 * 128, 131072],
                                       vtype=vtype, vr2c=(True,),
                                       dims_max=2, ndim_max=2, verbose=v,
                                       dry_run=dry_run, shuffle_axes=True,
                                       secondary_long_axis_size=3)
                    ct += tmp[0]
                    vkwargs += tmp[1]
                else:
                    self.run_fft_parallel(vkwargs)
                if dry_run and self.verbose:
                    print(f"Running {ct} R2C tests (backend: {self.backend_info(backend)})")

    @unittest.skipIf(not (has_pycuda or has_cupy or has_pyopencl), "No OpenCL/CUDA backend is available")
    @unittest.skipIf(not has_dct_ref, "scipy and pyfftw are not available - cannot test DCT")
    def test_dct(self):
        """Run DCT tests"""
        for backend in self.vbackend:
            init_ctx(backend, gpu_name=self.gpu, opencl_platform=self.opencl_platform, verbose=False)
            has_cl_fp64 = gpu_ctx_dic["pyopencl"][3] if backend == "pyopencl" else True
            ct = 0
            vkwargs = []
            for dry_run in [True, False]:
                vtype = (np.float32, np.float64)
                if backend == "pyopencl" and not has_cl_fp64:
                    vtype = (np.float32,)
                v = self.verbose and not dry_run
                if dry_run or self.nproc == 1:
                    tmp = self.run_fft([backend], [15, 17], dims_max=4, ndim_max=3,
                                       vtype=vtype, vdct=range(1, 5), vnorm=[1],
                                       verbose=v, dry_run=dry_run, shuffle_axes=True)
                    ct += tmp[0]
                    vkwargs += tmp[1]
                    # Avoid large array sizes for 30, 34
                    tmp = self.run_fft([backend], [30, 34], dims_max=4, ndim_max=4,
                                       vtype=vtype, vdct=range(1, 5), vnorm=[1],
                                       verbose=v, dry_run=dry_run, shuffle_axes=True,
                                       secondary_long_axis_size=3)
                    ct += tmp[0]
                    vkwargs += tmp[1]
                    # Larger 1D and 2D tests
                    tmp = self.run_fft([backend],
                                       [808, 2988, 4200, 8232, 8193, 8194,
                                        8232 * 128, 8193 * 128, 8194 * 128, 131072],
                                       vtype=vtype, dims_max=2, vdct=range(1, 5),
                                       vnorm=[1], verbose=v, dry_run=dry_run, shuffle_axes=True,
                                       secondary_long_axis_size=3)
                    ct += tmp[0]
                    vkwargs += tmp[1]
                else:
                    self.run_fft_parallel(vkwargs)
                if dry_run and self.verbose:
                    print(f"Running {ct} DCT tests (backend: {self.backend_info(backend)})")

    @unittest.skipIf(not (has_pycuda or has_cupy or has_pyopencl), "No OpenCL/CUDA backend is available")
    @unittest.skipIf(not has_dct_ref, "scipy and pyfftw are not available - cannot test DST")
    def test_dst(self):
        """Run DST tests"""
        for backend in self.vbackend:
            init_ctx(backend, gpu_name=self.gpu, opencl_platform=self.opencl_platform, verbose=False)
            has_cl_fp64 = gpu_ctx_dic["pyopencl"][3] if backend == "pyopencl" else True
            ct = 0
            vkwargs = []
            for dry_run in [True, False]:
                vtype = (np.float32, np.float64)
                if backend == "pyopencl" and not has_cl_fp64:
                    vtype = (np.float32,)
                v = self.verbose and not dry_run
                if dry_run or self.nproc == 1:
                    tmp = self.run_fft([backend], [15, 17], dims_max=4, ndim_max=4,
                                       vtype=vtype, vdst=range(1, 5),
                                       vnorm=[1], verbose=v, dry_run=dry_run, shuffle_axes=True)
                    ct += tmp[0]
                    vkwargs += tmp[1]
                    # Avoid large array sizes for 30, 34
                    tmp = self.run_fft([backend], [30, 34], dims_max=4, ndim_max=4,
                                       vtype=vtype, vdst=range(1, 5), vnorm=[1],
                                       verbose=v, dry_run=dry_run, shuffle_axes=True,
                                       secondary_long_axis_size=3)
                    ct += tmp[0]
                    vkwargs += tmp[1]
                    # Larger 1D and 2D tests
                    tmp = self.run_fft([backend],
                                       [808, 2988, 4200, 8232, 8193, 8194,
                                        8232 * 128, 8193 * 128, 8194 * 128, 131072],
                                       vtype=vtype, dims_max=2, vdst=range(1, 5),
                                       vnorm=[1], verbose=v, dry_run=dry_run, shuffle_axes=True,
                                       secondary_long_axis_size=3)
                    ct += tmp[0]
                    vkwargs += tmp[1]
                else:
                    self.run_fft_parallel(vkwargs)
                if dry_run and self.verbose:
                    print(f"Running {ct} DST tests (backend: {self.backend_info(backend)})")

    @unittest.skipIf(not has_pycuda, "pycuda is not available")
    def test_pycuda_streams(self):
        """
        Test multiple FFT in // with different cuda streams.
        """
        for dtype in (np.complex64, np.complex128):
            with self.subTest(dtype=np.dtype(dtype)):
                init_ctx("pycuda", gpu_name=self.gpu, opencl_platform=self.opencl_platform, verbose=False)
                if dtype == np.complex64:
                    rtol = 1e-6
                else:
                    rtol = 1e-12
                sh = (256, 256)
                d = (np.random.uniform(-0.5, 0.5, sh) + 1j * np.random.uniform(-0.5, 0.5, sh)).astype(dtype)
                n_streams = 5
                vd = []
                vapp = []
                for i in range(n_streams):
                    vd.append(cua.to_gpu(np.roll(d, i * 7, axis=1)))
                    vapp.append(cuVkFFTApp(d.shape, d.dtype, ndim=2, norm=1, stream=cu_drv.Stream()))

                for i in range(n_streams):
                    vapp[i].fft(vd[i])
                for i in range(n_streams):
                    dn = fftn(np.roll(d, i * 7, axis=1))
                    self.assertTrue(np.allclose(dn, vd[i].get(), rtol=rtol, atol=abs(dn).max() * rtol))

                for i in range(n_streams):
                    vapp[i].ifft(vd[i])
                for i in range(n_streams):
                    dn = np.roll(d, i * 7, axis=1)
                    self.assertTrue(np.allclose(dn, vd[i].get(), rtol=rtol, atol=abs(dn).max() * rtol))

    @unittest.skipIf(not has_pyopencl, "pyopencl is not available")
    def test_pyopencl_queues(self):
        """
        Test multiple FFT with queues different from the queue used
        in creating the VkFFTApp
        """
        init_ctx("pyopencl", gpu_name=self.gpu, opencl_platform=self.opencl_platform, verbose=False)
        has_cl_fp64 = gpu_ctx_dic["pyopencl"][3] if "pyopencl" in self.vbackend else True
        vtype = (np.complex64, np.complex128)
        if not has_cl_fp64:
            vtype = (np.complex64,)
        # Disable warning
        old_warning = pyvkfft.config.WARN_OPENCL_QUEUE_MISMATCH
        pyvkfft.config.WARN_OPENCL_QUEUE_MISMATCH = False
        for dtype in vtype:
            with self.subTest(dtype=np.dtype(dtype)):
                init_ctx("pyopencl", gpu_name=self.gpu, opencl_platform=self.opencl_platform, verbose=False)
                ctx = gpu_ctx_dic["pyopencl"][2].context
                if dtype == np.complex64:
                    rtol = 5e-6
                else:
                    rtol = 1e-12
                sh = (256, 256)
                d = (np.random.uniform(-0.5, 0.5, sh) + 1j * np.random.uniform(-0.5, 0.5, sh)).astype(dtype)
                n_queues = 5
                vd = []
                vapp = []

                for i in range(n_queues):
                    vapp.append(clVkFFTApp(d.shape, d.dtype, ndim=2, norm=1, queue=cl.CommandQueue(ctx)))

                queues = [cl.CommandQueue(ctx) for _ in range(2 * n_queues)]
                for i in range(n_queues):
                    vd.append(cla.to_device(queues[i], np.roll(d, i * 7, axis=1)))

                # test transforms with a supplied queue
                for i in range(n_queues):
                    vapp[i].fft(vd[i], queue=queues[i])
                    dn = fftn(np.roll(d, i * 7, axis=1))
                    self.assertTrue(np.allclose(dn, vd[i].get(), rtol=rtol, atol=abs(dn).max() * rtol))

                for i in range(n_queues):
                    vapp[i].ifft(vd[i], queue=queues[i])
                    dn = np.roll(d, i * 7, axis=1)
                    self.assertTrue(np.allclose(dn, vd[i].get(), rtol=rtol, atol=abs(dn).max() * rtol))

                # test transforms with a supplied queue, different from the arrays'
                for i in range(n_queues):
                    vapp[i].fft(vd[i], queue=queues[i + n_queues])
                    dn = fftn(np.roll(d, i * 7, axis=1))
                    queues[i + n_queues].finish()
                    self.assertTrue(np.allclose(dn, vd[i].get(), rtol=rtol, atol=abs(dn).max() * rtol))

                for i in range(n_queues):
                    vapp[i].ifft(vd[i], queue=queues[i + n_queues])
                    dn = np.roll(d, i * 7, axis=1)
                    queues[i + n_queues].finish()
                    self.assertTrue(np.allclose(dn, vd[i].get(), rtol=rtol, atol=abs(dn).max() * rtol))

                # test transforms with the arrays queues
                for i in range(n_queues):
                    vapp[i].fft(vd[i])
                    dn = fftn(np.roll(d, i * 7, axis=1))
                    self.assertTrue(np.allclose(dn, vd[i].get(), rtol=rtol, atol=abs(dn).max() * rtol))

                for i in range(n_queues):
                    vapp[i].ifft(vd[i])
                    dn = np.roll(d, i * 7, axis=1)
                    self.assertTrue(np.allclose(dn, vd[i].get(), rtol=rtol, atol=abs(dn).max() * rtol))

                # Test using the simple fft interface: inplace, same queue as array
                for i in range(n_queues):
                    vd[i] = vkfftn(vd[i], vd[i])
                    dn = fftn(np.roll(d, i * 7, axis=1))
                    self.assertTrue(np.allclose(dn, vd[i].get(), rtol=rtol, atol=abs(dn).max() * rtol))

                    vd[i] = vkifftn(vd[i], vd[i])
                    dn = np.roll(d, i * 7, axis=1)
                    self.assertTrue(np.allclose(dn, vd[i].get(), rtol=rtol, atol=abs(dn).max() * rtol))

                # Test using the simple fft interface: inplace, different queue for the transform
                # A synchronisation is needed as get() will use the array's queue and not the transform's
                for i in range(n_queues):
                    vd[i] = vkfftn(vd[i], vd[i], cl_queue=queues[i + n_queues])
                    dn = fftn(np.roll(d, i * 7, axis=1))
                    queues[i + n_queues].finish()
                    self.assertTrue(np.allclose(dn, vd[i].get(), rtol=rtol, atol=abs(dn).max() * rtol))

                    vd[i] = vkifftn(vd[i], vd[i], cl_queue=queues[i + n_queues])
                    dn = np.roll(d, i * 7, axis=1)
                    queues[i + n_queues].finish()
                    self.assertTrue(np.allclose(dn, vd[i].get(), rtol=rtol, atol=abs(dn).max() * rtol))

                # Test using the simple fft interface: out-of-place, same queue as array
                # The newly allocated array has the same queu as the source array, so no synchronisation issue
                for i in range(n_queues):
                    d1 = vkfftn(vd[i])
                    dn = fftn(np.roll(d, i * 7, axis=1))
                    self.assertTrue(np.allclose(dn, d1.get(), rtol=rtol, atol=abs(dn).max() * rtol))

                    d2 = vkifftn(d1)
                    dn = np.roll(d, i * 7, axis=1)
                    self.assertTrue(np.allclose(dn, d2.get(), rtol=rtol, atol=abs(dn).max() * rtol))

                # Test using the simple fft interface: out-of-place, different queue for the transform
                # A synchronisation is needed as get() will use the array's queue and not the transform's
                for i in range(n_queues):
                    d1 = vkfftn(vd[i], cl_queue=queues[i + n_queues])
                    dn = fftn(np.roll(d, i * 7, axis=1))
                    queues[i + n_queues].finish()
                    self.assertTrue(np.allclose(dn, d1.get(), rtol=rtol, atol=abs(dn).max() * rtol))

                    d2 = vkifftn(d1, cl_queue=queues[i + n_queues])
                    dn = np.roll(d, i * 7, axis=1)
                    queues[i + n_queues].finish()
                    self.assertTrue(np.allclose(dn, d2.get(), rtol=rtol, atol=abs(dn).max() * rtol))
        pyvkfft.config.WARN_OPENCL_QUEUE_MISMATCH = old_warning

    @unittest.skipIf(not has_pycuda, "pycuda is not available")
    @unittest.skipIf(len(v_gpu_pycuda) < 2, f"{len(v_gpu_pycuda)}<2 devices available")
    def test_zz_multi_gpu_fft_threads_pycuda(self):
        """
        Test the pyvkfft.fft (cached) interface with multiple GPU and parallel threads with pycuda streams
        """
        # Notes:
        # * this only works with pycuda if we supply a stream for each transform
        # * this gives a warning: "device_allocation in out-of-thread context could not be cleaned up"

        clear_vkfftapp_cache()

        def thread_fn(dev):
            cu_ctx = dev.make_context()
            cu_ctx.push()
            # Note:
            s = cu_drv.Stream()
            for i in range(5):  # try to make sure threads will exec in //
                gpu_data = cua.to_gpu(np.ones(2 ** 16).astype(np.complex64))
                fft_gpu = vkfftn(gpu_data, cuda_stream=s)
                time.sleep(0.01)
            cu_ctx.pop()
            cu_ctx.detach()

        threads = []
        for d in v_gpu_pycuda:
            t = threading.Thread(target=thread_fn, args=(d,))
            # t.daemon = True
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # Multi-GPU cupy test modify the cuda context so are executed last (hence the zz),
    # to avoid messing with pycuda context management

    @unittest.skipIf(not has_cupy, "cupy is not available")
    @unittest.skipIf(len(v_gpu_cupy) < 2, f"{len(v_gpu_cupy)}<2 devices available")
    def test_zz_multi_gpu_fft_threads_cupy(self):
        """
        Test the pyvkfft.fft (cached) interface with multiple GPU and parallel threads with cupy
        """
        clear_vkfftapp_cache()

        def thread_fn(dev):
            with dev.use():
                for i in range(5):  # try to make sure threads will exec in //
                    gpu_data = cp.array(np.ones(2 ** 16).astype(np.complex64))
                    fft_gpu = vkfftn(gpu_data)
                    time.sleep(0.01)

        threads = []
        for d in v_gpu_cupy:
            t = threading.Thread(target=thread_fn, args=(d,))
            # t.daemon = True
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    @unittest.skipIf(not has_pyopencl, "pyopencl is not available")
    @unittest.skipIf(len(v_gpu_pyopencl) < 2, f"{len(v_gpu_pyopencl)}<2 devices available")
    def test_zz_multi_gpu_fft_threaded_pyopencl(self):
        """
        Test the pyvkfft.fft (cached) interface with multiple GPU and parallel threads with pyopencl
        """
        clear_vkfftapp_cache()

        def thread_fn(dev):
            cl_ctx = cl.Context([dev])
            cq = cl.CommandQueue(cl_ctx)

            for i in range(5):  # try to make sure threads will exec in //
                gpu_data = cla.to_device(cq, np.ones(2 ** 16).astype(np.complex64))
                fft_gpu = vkfftn(gpu_data)
                time.sleep(0.01)

        threads = []
        for d in v_gpu_pyopencl:
            t = threading.Thread(target=thread_fn, args=(d,))
            # t.daemon = True
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()


# The class parameters are written in pyvkfft_test.main()
class TestFFTSystematic(unittest.TestCase):
    axes = None
    bluestein = False
    colour = False
    dct = False
    dst = False
    db = None
    dry_run = False
    dtype = np.float32
    fast_random = None
    graph = None
    gpu = None
    inplace = False
    lut = False
    fstride = False
    max_pow = None
    max_nb_tests = 1000
    nb_test = 0  # Number of tests actually run
    nb_shapes_gen = None
    ndim = 1
    # t.ndims = args.ndims
    norm = 1
    nproc = 1
    opencl_platform = None
    r2c = False
    radix = None
    range = 2, 128
    range_nd_narrow = 0, 0
    range_size = 0, 128 * 1024 ** 2 // 8
    ref_long_double = False
    serial = False
    timeout = 30
    vbackend = None
    verbose = True
    vshape = []

    def setUp(self) -> None:
        if self.vbackend is None:
            self.vbackend = []
            if has_pycuda:
                self.vbackend.append("pycuda")
            if has_cupy:
                self.vbackend.append("cupy")
            if has_pyopencl:
                self.vbackend.append("pyopencl")
                init_ctx("pyopencl", gpu_name=self.gpu, opencl_platform=self.opencl_platform, verbose=False)
                self.cq, self.has_cl_fp64 = gpu_ctx_dic["pyopencl"][2:]
        self.assertTrue(not self.bluestein or self.radix is None, "Cannot select both Bluestein and radix")
        r2r = self.dct if self.dct else self.dst  # special case for radix generation (DCT/DST 1 and 4)
        if not self.bluestein and self.radix is None:
            self.vshape = radix_gen_n(nmax=self.range[1], max_size=self.range_size[1], radix=None,
                                      ndim=self.ndim, nmin=self.range[0], max_pow=self.max_pow,
                                      range_nd_narrow=self.range_nd_narrow, min_size=self.range_size[0])
        elif self.bluestein:
            self.vshape = radix_gen_n(nmax=self.range[1], max_size=self.range_size[1],
                                      radix=(2, 3, 5, 7, 11, 13), ndim=self.ndim,
                                      inverted=True, nmin=self.range[0], max_pow=self.max_pow,
                                      range_nd_narrow=self.range_nd_narrow, min_size=self.range_size[0],
                                      r2r=r2r)
        else:
            if len(self.radix) == 0:
                self.radix = [2, 3, 5, 7, 11, 13]
            if self.r2c and 2 not in self.radix:  # and inplace ?
                raise RuntimeError("For r2c, the x/fastest axis must be even (requires radix-2)")
            self.vshape = radix_gen_n(nmax=self.range[1], max_size=self.range_size[1],
                                      radix=self.radix, ndim=self.ndim,
                                      nmin=self.range[0], max_pow=self.max_pow,
                                      range_nd_narrow=self.range_nd_narrow, min_size=self.range_size[0],
                                      r2r=r2r)
        if not self.dry_run:
            self.assertTrue(len(self.vshape), "The list of sizes to test is empty !")
            if self.max_nb_tests:
                self.assertTrue(len(self.vshape) <= self.max_nb_tests, "Too many array shapes have been generated: "
                                                                       "%d > %d [parameter hint: max-nb-tests]" %
                                (len(self.vshape), self.max_nb_tests))

    def test_systematic(self):
        if self.dry_run:
            # The array shapes to test have been generated
            if self.verbose:
                print("Dry run: %d array shapes generated" % len(self.vshape))
            # OK, this lacks elegance, but works to get back the value in the scripts
            self.__class__.nb_shapes_gen = len(self.vshape)
            return
        # Generate the list of configurations as kwargs for test_accuracy()
        vkwargs = []
        for backend in self.vbackend:
            for s in self.vshape:
                if self.fast_random is not None and len(vkwargs):
                    # Randomly skip tests to go faster
                    if np.random.uniform(0, 100) > self.fast_random:
                        continue
                kwargs = {"backend": backend, "shape": s, "ndim": len(s), "axes": self.axes,
                          "dtype": self.dtype, "inplace": self.inplace, "norm": self.norm, "use_lut": self.lut,
                          "r2c": self.r2c, "dct": self.dct, "dst": self.dst, "gpu_name": self.gpu,
                          "opencl_platform": self.opencl_platform, "stream": None, "verbose": False,
                          "colour_output": self.colour, "ref_long_double": self.ref_long_double,
                          "order": 'F' if self.fstride else 'C'}
                vkwargs.append(kwargs)
        if self.db is not None:
            # TODO secure the db with a context 'with'
            db = sqlite3.connect(self.db)
            dbc = db.cursor()
            dbc.execute('CREATE TABLE IF NOT EXISTS pyvkfft_test (epoch int, hostname int,'
                        'backend text, language text, transform text, axes text, array_shape text,'
                        'ndims int, ndim int, precision int, inplace int, norm int, lut int, fstride int,'
                        'n int, n2_fft float, n2_ifft float, ni_fft float, ni_ifft float, tolerance float,'
                        'dt_app float, dt_fft float, dt_ifft float, src_unchanged_fft int, src_unchanged_ifft int,'
                        'gpu_name text, success int, error int, vkfft_error_code int)')
            db.commit()
            hostname = socket.gethostname()
            lang = 'opencl' if 'opencl' in backend else 'cuda'
            if self.r2c:
                transform = "R2C"
            elif self.dct:
                transform = "DCT%d" % self.dct
            elif self.dst:
                transform = "DST%d" % self.dst
            else:
                transform = "C2C"

        # For graph output
        vn, vni, vn2, vnii, vn2i, vblue, vshape = [], [], [], [], [], [], []
        gpu_name = "GPU"

        if self.verbose:
            print("Starting %d tests..." % (len(vkwargs)))
        t0 = timeit.default_timer()

        # Handle timeouts if for some weird reason a process hangs indefinitely
        nb_timeout = 0
        i_start = 0

        while True:
            timeout = False
            # Need to use spawn to handle the GPU context
            with multiprocessing.get_context('spawn').Pool(self.nproc) as pool:
                if not self.serial:
                    results = pool.imap(test_accuracy_kwargs, vkwargs[i_start:], chunksize=1)
                for i in range(i_start, len(vkwargs)):
                    v = vkwargs[i]
                    sh = v['shape']
                    ndim = len(sh)
                    # We use np.dtype(dtype) instead of dtype because it is written out simply
                    # as e.g. "float32" instead of "<class 'numpy.float32'>"
                    with self.subTest(backend=backend, shape=sh, ndim=ndim,
                                      dtype=np.dtype(self.dtype), norm=self.norm, use_lut=self.lut,
                                      inplace=self.inplace, r2c=self.r2c,
                                      dct=self.dct, dst=self.dst, fstride=self.fstride):
                        if self.serial:
                            res = test_accuracy_kwargs(v)
                        else:
                            try:
                                # increase timeout for large 3D systems
                                r = 2 if np.prod(sh) > 2 ** 24 and ndim == 3 else 1
                                res = results.next(timeout=self.timeout * r)
                            except multiprocessing.TimeoutError as ex:
                                # NB: the timeout won't change the next() result, so will need
                                # to terminate & restart the pool
                                timeout = True
                                raise ex
                        n = max(res['shape'])
                        npr = primes(n)
                        ni, n2 = res["ni"], res["n2"]
                        nii, n2i = res["nii"], res["n2i"]
                        tol = res["tol"]
                        src1 = res["src_unchanged_fft"]
                        src2 = res["src_unchanged_ifft"]
                        succ = max(ni, nii) < tol

                        vn.append(n)
                        vblue.append(max(npr) > 13)
                        vni.append(ni)
                        vn2.append(n2)
                        vn2i.append(n2i)
                        vnii.append(nii)
                        vshape.append(sh)
                        if len(vn) == 1:
                            gpu_name = res["gpu_name"]

                        if not self.inplace:
                            if not src1:
                                succ = False
                            elif not self.r2c and not src2:
                                succ = False
                        if self.db is not None:
                            dbc.execute('INSERT INTO pyvkfft_test VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'
                                        '?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                                        (time.time(), hostname, backend, lang, transform,
                                         str(res['axes']).encode('ascii'), str(res['shape']).encode('ascii'),
                                         len(res['shape']), ndim, np.dtype(self.dtype).itemsize,
                                         self.inplace, self.norm, self.lut, self.fstride,
                                         int(max(res['shape'])), float(n2), float(n2i),
                                         float(ni), float(nii), float(tol), res["dt_app"], res["dt_fft"],
                                         res["dt_ifft"],
                                         int(src1), int(src2), res["gpu_name"].encode('ascii'), int(succ), 0, 0))
                            db.commit()
                        if self.verbose:
                            print(res['str'])
                        self.assertTrue(ni < tol, "Accuracy mismatch after FFT, n2=%8e ni=%8e>%8e" % (n2, ni, tol))
                        self.assertTrue(nii < tol, "Accuracy mismatch after iFFT, n2=%8e ni=%8e>%8e" % (n2, nii, tol))
                        if not self.inplace:
                            self.assertTrue(src1, "The source array was modified during the FFT")
                            if platform.system() == 'Darwin':
                                nmaxr2c1d = 2048 * (1 + int(self.dtype in (np.float32, np.complex64)))
                            else:
                                nmaxr2c1d = 3072 * (1 + int(self.dtype in (np.float32, np.complex64)))
                            if not self.r2c or (ndim == 1 and max(npr) <= 13) and n < nmaxr2c1d:
                                # Only 1D radix C2R do not alter the source array, if n<=?
                                self.assertTrue(src2,
                                                "The source array was modified during the iFFT %d %d" % (n, nmaxr2c1d))
                    if timeout:
                        # One process is stuck, must kill the pool and start again
                        if self.verbose:
                            print("Timeout for N=%d. Re-starting the pool..." % max(v['shape']))
                        i_start = i + 1
                        pool.terminate()
                        nb_timeout += 1
                        break
            if not timeout or i_start >= len(vkwargs) or nb_timeout >= 4:
                break
        self.__class__.nb_test = len(self.vbackend) * len(vkwargs)
        if self.verbose:
            print("Finished %d tests in %s" %
                  (len(vkwargs), time.strftime("%Hh %Mm %Ss", time.gmtime(timeit.default_timer() - t0))))

        if self.graph is not None and len(vn):
            if self.r2c:
                t = "R2C"
            elif self.dct:
                t = "DCT%d" % self.dct
            elif self.dst:
                t = "DST%d" % self.dst
            else:
                t = "C2C"

            tmp = ""
            if self.lut:
                tmp += "_lut"
            if self.inplace:
                tmp += "_inplace"

            r = ""
            if self.radix is not None:
                r = "_radix"
                for k in self.radix:
                    r += "-%d" % k
            elif self.bluestein:
                r = "_bluestein"

            vkfft_ver = f'{vkfft_version()}' if 'unknown' in vkfft_git_version() \
                else f'{vkfft_version()}[{vkfft_git_version()}]'
            tit = "%s %s pyvkfft %s VkFFT %s" % (gpu_name, self.vbackend[0], __version__, vkfft_ver)
            if self.ndim == 12:
                sndim = "1D2D"
            elif self.ndim == 123:
                sndim = "1D2D3D"
            else:
                sndim = "%dD" % self.ndim
            suptit = " %s %s%s N=%d-%d norm=%d %s%s" % \
                     (t, sndim, r, self.range[0], self.range[1], self.norm, str(np.dtype(np.float32)), tmp)
            if self.ref_long_double and has_scipy:
                suptit += " [long double ref]"
            suptit += " [%d tests]" % self.nb_test

            import matplotlib as mpl
            mpl.use('Agg')
            import matplotlib.pyplot as plt
            from scipy import stats
            plt.figure(figsize=(8, 5))

            x = np.array([np.prod(s) for s in vshape], dtype=np.float32)
            xl = np.log10(x)
            ms = 4
            plt.semilogx(x, vni, 'ob', label=r"$[FFT]L_{\infty}$", alpha=0.2, ms=ms)
            plt.semilogx(x, vnii, 'og', label=r"$[IFFT]L_{\infty}$", alpha=0.2, ms=ms)

            r2 = stats.linregress(xl, np.array(vn2, dtype=np.float32))
            plt.semilogx(x, vn2, "^b", ms=ms,
                         label=r"$[FFT]L2\approx %s+%s\log(size)$" % (latex_float(r2[1]), latex_float(r2[0])))

            r2i = stats.linregress(xl, np.array(vn2i, dtype=np.float32))
            plt.semilogx(x, vn2, "vg", ms=ms,
                         label=r"$[IFFT]L2\approx %s+%s\log(size)$" % (latex_float(r2i[1]), latex_float(r2i[0])))

            plt.semilogx(x, r2[1] + r2[0] * xl, "b-")
            plt.semilogx(x, r2i[1] + r2i[0] * xl, "g-")
            plt.title(tit.replace('_', ' '), fontsize=10)
            plt.suptitle(suptit, fontsize=12)
            plt.grid(True)
            plt.legend(loc='upper left')
            plt.xlabel("size", loc='right')
            plt.tight_layout()
            graph = self.graph
            if not len(graph):
                graph = "%s_%s_%s_%s%s_%d-%d_norm%d_%s%s.svg" % \
                        (gpu_name.replace(' ', ''), self.vbackend[0], t, sndim, r, self.range[0],
                         self.range[1], self.norm, str(np.dtype(np.float32)), tmp)
            plt.savefig(graph)
            if self.verbose:
                print("Saved accuracy graph to: %s" % graph)

        if nb_timeout >= 4:
            raise RuntimeError("4 multiprocessing timeouts while testing... giving up")


def suite():
    test_suite = unittest.TestSuite()
    load_tests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(load_tests(TestFFT))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite', verbosity=2)
