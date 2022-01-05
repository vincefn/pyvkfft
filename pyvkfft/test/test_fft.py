# -*- coding: utf-8 -*-

# PyVkFFT
#   (c) 2021- : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr
#
#
# pyvkfft unit tests.

import os
import sys
import unittest
import multiprocessing
import numpy as np

try:
    from scipy.misc import ascent
except ImportError:
    def ascent():
        return np.random.randint(0, 255, (512, 512))

from pyvkfft.base import primes, radix_gen
from pyvkfft.fft import fftn as vkfftn, ifftn as vkifftn, rfftn as vkrfftn, \
    irfftn as vkirfftn, dctn as vkdctn, idctn as vkidctn
from pyvkfft.accuracy import test_accuracy, test_accuracy_kwargs, exhaustive_test, fftn, cq, has_cl_fp64

try:
    import pycuda.autoinit
    import pycuda.gpuarray as cua

    import pycuda.driver as cu_drv
    from pyvkfft.cuda import VkFFTApp as cuVkFFTApp

    has_pycuda = True
except ImportError:
    has_pycuda = False

try:
    import cupy as cp

    # TODO: The following somehow helps initialising cupy, not sure why it's useful.
    #  (some context auto-init...). Otherwise a cuLaunchKernel error occurs with
    #  the first transform.
    cupy_a = cp.array(np.zeros((128, 128), dtype=np.float32))
    cupy_a.sum()

    has_cupy = True
except ImportError:
    has_cupy = False

try:
    import pyopencl as cl
    import pyopencl.array as cla

    has_pyopencl = True
except ImportError:
    has_pyopencl = False


class TestFFT(unittest.TestCase):

    def test_backend(self):
        self.assertTrue(has_pycuda or has_pyopencl or has_cupy,
                        "Either pycuda, pyopencl or cupy must be available")

    @unittest.skipIf(not (has_pycuda or has_cupy or has_pyopencl), "No OpenCL/CUDA backend is available")
    def test_simple_fft(self):
        """Test the simple fft API"""
        vbackend = []
        if has_pycuda:
            vbackend.append("pycuda")
        if has_cupy:
            vbackend.append("cupy")
        if has_pyopencl:
            vbackend.append("pyopencl")

        for backend in vbackend:
            if backend == "pycuda":
                dc = cua.to_gpu(ascent().astype(np.complex64))
                dr = cua.to_gpu(ascent().astype(np.float32))
            elif backend == "cupy":
                dc = cp.array(ascent().astype(np.complex64))
                dr = cp.array(ascent().astype(np.float32))
            else:
                dc = cla.to_device(cq, ascent().astype(np.complex64))
                dr = cla.to_device(cq, ascent().astype(np.float32))
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
            d = vkirfftn(d)

            # DCT, new destination array
            d = vkdctn(dr)
            d = vkidctn(d)
            # DCT, out-of-place
            d2 = dr.copy()
            d2 = vkdctn(dr, d2)
            dr = vkidctn(d2, dr)

    def run_fft(self, vbackend, vn, dims_max=4, ndim_max=3, shuffle_axes=True,
                vtype=(np.complex64, np.complex128),
                vlut="auto", vinplace=(True, False), vnorm=(0, 1),
                vr2c=(False,), vdct=(False,), verbose=False, dry_run=False):
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
        :param verbose: True or False - prints two lines per test (FFT and iFFT result)
        :param dry_run: if True, only count the number of test to run
        :return: the number of tests performed
        """
        ct = 0
        for backend in vbackend:
            for n in vn:
                for dims in range(1, dims_max + 1):
                    for ndim0 in range(1, min(dims, ndim_max) + 1):
                        for r2c in vr2c:
                            for dct in vdct:
                                # Setup use of either ndim or axes, also test skipping dimensions
                                ndim_axes = [(ndim0, None)]
                                if shuffle_axes and not (r2c or dct):
                                    for i in range(1, 2 ** (ndim0 - 1)):
                                        axes = []
                                        for ii in range(ndim0):
                                            if not (i & 2 ** ii):
                                                axes.append(-ii - 1)
                                        ndim_axes.append((None, axes))
                                for ndim, axes in ndim_axes:
                                    for dtype in vtype:
                                        if axes is None:
                                            axes_numpy = list(range(dims))[-ndim:]
                                        else:
                                            axes_numpy = axes

                                        # Array shape
                                        sh = [n] * dims

                                        # Use only a size of 2 for non-transform axes
                                        for ii in range(len(sh)):
                                            if ii not in axes_numpy and (-len(sh) + ii) not in axes_numpy:
                                                sh[ii] = 2
                                        if not dry_run:
                                            if dtype in (np.float32, np.float64):
                                                d0 = np.random.uniform(-0.5, 0.5, sh).astype(dtype)
                                            else:
                                                d0 = (np.random.uniform(-0.5, 0.5, sh)
                                                      + 1j * np.random.uniform(-0.5, 0.5, sh)).astype(dtype)
                                        if vlut == "auto":
                                            if dtype in (np.float64, np.complex128):
                                                # By default LUT is enabled for complex128, no need to test twice
                                                tmp = [None]
                                            else:
                                                tmp = [None, True]
                                        else:
                                            tmp = vlut
                                        for use_lut in tmp:
                                            for inplace in vinplace:
                                                for norm in vnorm:
                                                    with self.subTest(backend=backend, n=n, dims=dims, ndim=ndim,
                                                                      axes=axes, dtype=dtype, norm=norm,
                                                                      use_lut=use_lut, inplace=inplace,
                                                                      r2c=r2c, dct=dct):
                                                        ct += 1
                                                        if not dry_run:
                                                            res = test_accuracy(backend, sh, ndim, axes, dtype, inplace,
                                                                                norm, use_lut, r2c=r2c, dct=dct,
                                                                                stream=None, queue=cq,
                                                                                return_array=False, init_array=d0,
                                                                                verbose=verbose)
                                                            ni, n2 = res["ni"], res["n2"]
                                                            nii, n2i = res["nii"], res["n2i"]
                                                            tol = res["tol"]
                                                            src1 = res["src_unchanged_fft"]
                                                            src2 = res["src_unchanged_ifft"]
                                                            self.assertTrue(ni < tol, "Accuracy mismatch after FFT, "
                                                                                      "n2=%8e ni=%8e>%8e" %
                                                                            (n2, ni, tol))
                                                            self.assertTrue(nii < tol, "Accuracy mismatch after iFFT, "
                                                                                       "n2=%8e ni=%8e>%8e" %
                                                                            (n2, nii, tol))
                                                            if not inplace:
                                                                self.assertTrue(src1, "The source array was modified "
                                                                                      "during the FFT")
                                                                if not r2c:
                                                                    self.assertTrue(src2,
                                                                                    "The source array was modified "
                                                                                    "during the iFFT")
        return ct

    @unittest.skipIf(not (has_pycuda or has_cupy or has_pyopencl), "No OpenCL/CUDA backend is available")
    def test_c2c(self):
        """Run C2C tests"""
        vbackend = []
        if has_pycuda:
            vbackend.append("pycuda")
        if has_cupy:
            vbackend.append("cupy")
        if has_pyopencl:
            vbackend.append("pyopencl")
        ct = 0
        for dry_run in [True, False]:
            for backend in vbackend:
                vtype = (np.complex64, np.complex128)
                if backend == "pyopencl" and not has_cl_fp64:
                    vtype = (np.complex64,)
                v = self.verbose and not dry_run
                ct += self.run_fft([backend], [30, 34], vtype=vtype, verbose=v, dry_run=dry_run)
                ct += self.run_fft([backend], [808], vtype=vtype, dims_max=2, verbose=v, dry_run=dry_run)
            if dry_run and self.verbose:
                print("Running %d C2C tests" % ct)

    @unittest.skipIf(not (has_pycuda or has_cupy or has_pyopencl), "No OpenCL/CUDA backend is available")
    def test_r2c(self):
        """Run R2C tests"""
        vbackend = []
        if has_pycuda:
            vbackend.append("pycuda")
        if has_cupy:
            vbackend.append("cupy")
        if has_pyopencl:
            vbackend.append("pyopencl")
        ct = 0
        for dry_run in [True, False]:
            for backend in vbackend:
                vtype = (np.float32, np.float64)
                if backend == "pyopencl" and not has_cl_fp64:
                    vtype = (np.float32,)
                v = self.verbose and not dry_run
                ct += self.run_fft([backend], [30, 34], vtype=vtype, vr2c=(True,), verbose=v, dry_run=dry_run)
                ct += self.run_fft([backend], [808], vtype=vtype, dims_max=2, vr2c=(True,), verbose=v, dry_run=dry_run)
            if dry_run and self.verbose:
                print("Running %d R2C tests" % ct)

    @unittest.skipIf(not (has_pycuda or has_cupy or has_pyopencl), "No OpenCL/CUDA backend is available")
    def test_dct(self):
        """Run DCT tests"""
        vbackend = []
        if has_pycuda:
            vbackend.append("pycuda")
        if has_cupy:
            vbackend.append("cupy")
        if has_pyopencl:
            vbackend.append("pyopencl")
        ct = 0
        for dry_run in [True, False]:
            for backend in vbackend:
                vtype = (np.float32, np.float64)
                if backend == "pyopencl" and not has_cl_fp64:
                    vtype = (np.float32,)
                v = self.verbose and not dry_run
                ct += self.run_fft([backend], [30, 34], vtype=vtype, vnorm=[1], vdct=range(1, 5), verbose=v,
                                   dry_run=dry_run)
            if dry_run and self.verbose:
                print("Running %d DCT tests" % ct)

    @unittest.skipIf(not has_pycuda, "pycuda is not available")
    def test_pycuda_streams(self):
        """
        Test multiple FFT in // with different cuda streams.
        """
        for dtype in (np.complex64, np.complex128):
            with self.subTest(dtype=dtype):
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


class TestFFTSystematic(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls) -> None:
    #     cls.axes = None
    #     cls.dct = False
    #     cls.bluestein = False
    #     cls.inplace = False
    #     cls.db = False
    #     cls.dry_run = False
    #     cls.dtype = np.float32
    #     cls.lut = None
    #     cls.ndim = 1
    #     cls.ndims = None
    #     cls.norm = 1
    #     cls.nproc = 1
    #     cls.r2c = False
    #     cls.radix = None
    #     cls.range = 2, 128
    #     cls.vbackend = None
    #     cls.verbose = False
    #     cls.vn = None

    def setUp(self) -> None:
        if self.vbackend is None:
            self.vbackend = []
            if has_pycuda:
                self.vbackend.append("pycuda")
            if has_cupy:
                self.vbackend.append("cupy")
            if has_pyopencl:
                self.vbackend.append("pyopencl")
        self.assertTrue(not self.bluestein or self.radix is None, "Cannot select both Bluestein and radix")
        if not self.bluestein and self.radix is None:
            self.vn = range(self.range[0], self.range[1] + 1)
        else:
            if self.r2c and 2 not in self.radix:  # and inplace ?
                raise RuntimeError("For r2c, the x/fastest axis must be even (requires radix-2)")
            if self.bluestein:
                self.vn = radix_gen(self.range[1], (2, 3, 5, 7, 11, 13), even=self.r2c,
                                    inverted=True, nmin=self.range[0])
            else:
                if len(self.radix) == 0:
                    self.radix = [2, 3, 5, 7, 11, 13]
                self.vn = radix_gen(self.range[1], self.radix, even=self.r2c, nmin=self.range[0])
        self.assertTrue(len(self.vn), "The list of sizes to test is empty !")

    def run_systematic(self, backend, vn, ndim, dtype, inplace, norm, use_lut, r2c=False, dct=False, nproc=None,
                       verbose=False):
        """
        Run tests on a large range of sizes using multiprocessing

        :param backend: either 'pyopencl', 'pycuda' or 'cupy'
        :param vn: the list/iterable of sizes n.
        :param ndim: the number of dimensions. The array shape will be [n]*ndim
        :param dtype: either np.complex64 or np.complex128, or np.float32/np.float64 for r2c & dct
        :param inplace: True or False
        :param norm: either 0, 1 or "ortho"
        :param use_lut: if True,1, False or 0, will trigger useLUT=1 or 0 for VkFFT.
            If None, the default VkFFT behaviour is used. Always True by default
            for double precision, so no need to force it.
        :param r2c: if True, test an r2c transform. If inplace, the last dimension
            (x, fastest axis) must be even
        :param dct: either 1, 2, 3 or 4 to test different dct. Only norm=1 is can be
            tested (native scipy/pyfftw normalisation).
        :param nproc: the maximum number of parallel process to use. If None, the
            number of detected cores will be used (this may use too much memory !)
        :return: nothing
        """
        # Generate the list of configurations as kwargs for test_accuracy()
        vkwargs = []
        for n in vn:
            kwargs = {"backend": backend, "shape": [n] * ndim, "ndim": ndim, "axes": None, "dtype": dtype,
                      "inplace": inplace, "norm": norm, "use_lut": use_lut, "r2c": r2c, "dct": dct, "stream": None,
                      "verbose": False}
            vkwargs.append(kwargs)
        # Need to use spawn to handle the GPU context
        with multiprocessing.get_context('spawn').Pool(nproc) as pool:
            for res in pool.imap(test_accuracy_kwargs, vkwargs):
                with self.subTest(backend=backend, n=n, ndim=ndim, dtype=dtype, norm=norm,
                                  use_lut=use_lut, inplace=inplace, r2c=r2c, dct=dct):
                    if verbose:
                        print(res['str'])
                    ni, n2 = res["ni"], res["n2"]
                    nii, n2i = res["nii"], res["n2i"]
                    tol = res["tol"]
                    src1 = res["src_unchanged_fft"]
                    src2 = res["src_unchanged_ifft"]
                    self.assertTrue(ni < tol, "Accuracy mismatch after FFT, n2=%8e ni=%8e>%8e" % (n2, ni, tol))
                    self.assertTrue(nii < tol, "Accuracy mismatch after iFFT, n2=%8e ni=%8e>%8e" % (n2, nii, tol))
                    if not inplace:
                        self.assertTrue(src1, "The source array was modified during the FFT")
                        if not r2c:
                            self.assertTrue(src2, "The source array was modified during the iFFT")

    def test_systematic(self):
        # Generate the list of configurations as kwargs for test_accuracy()
        vkwargs = []
        for backend in self.vbackend:
            for n in self.vn:
                kwargs = {"backend": backend, "shape": [n] * self.ndim, "ndim": self.ndim, "axes": self.axes,
                          "dtype": self.dtype, "inplace": self.inplace, "norm": self.norm, "use_lut": self.lut,
                          "r2c": self.r2c, "dct": self.dct, "stream": None, "verbose": False,
                          "colour_output": self.colour}
                vkwargs.append(kwargs)
        # Need to use spawn to handle the GPU context
        with multiprocessing.get_context('spawn').Pool(self.nproc) as pool:
            for res in pool.imap(test_accuracy_kwargs, vkwargs):
                with self.subTest(backend=backend, n=max(res['shape']), ndim=self.ndim,
                                  dtype=self.dtype, norm=self.norm, use_lut=self.lut,
                                  inplace=self.inplace, r2c=self.r2c, dct=self.dct):
                    if self.verbose:
                        print(res['str'])
                    ni, n2 = res["ni"], res["n2"]
                    nii, n2i = res["nii"], res["n2i"]
                    tol = res["tol"]
                    src1 = res["src_unchanged_fft"]
                    src2 = res["src_unchanged_ifft"]
                    self.assertTrue(ni < tol, "Accuracy mismatch after FFT, n2=%8e ni=%8e>%8e" % (n2, ni, tol))
                    self.assertTrue(nii < tol, "Accuracy mismatch after iFFT, n2=%8e ni=%8e>%8e" % (n2, nii, tol))
                    if not self.inplace:
                        self.assertTrue(src1, "The source array was modified during the FFT")
                        if not self.r2c:
                            self.assertTrue(src2, "The source array was modified during the iFFT")

    def _test_systematic_c2c(self):
        """Systematic C2C tests, without shuffling axes"""
        vbackend = []
        if has_pycuda:
            vbackend.append("pycuda")
        if has_cupy:
            vbackend.append("cupy")
        if has_pyopencl:
            vbackend.append("pyopencl")
        for backend in vbackend:
            vtype = (np.float32, np.float64)
            if backend == "pyopencl" and not has_cl_fp64:
                vtype = (np.float32,)
            for dtype in vtype:
                vlut = [None]
                if dtype == np.float32:
                    vlut += [True]
                for inplace in [True, False]:
                    for norm in [0, 1]:
                        for lut in vlut:
                            self.run_exhaustive(backend, range(2, 15000), ndim=1, dtype=dtype, inplace=inplace,
                                                norm=norm, use_lut=lut, nproc=16, verbose=True)
                            self.run_exhaustive(backend, range(2, 4500), ndim=2, dtype=dtype, inplace=inplace,
                                                norm=norm, use_lut=lut, nproc=16, verbose=True)
                            self.run_exhaustive(backend, range(2, 550), ndim=3, dtype=dtype, inplace=inplace,
                                                norm=norm, use_lut=lut, nproc=4, verbose=True)

    def _test_systematic_r2c(self):
        """Systematic R2C tests, without shuffling axes"""
        vbackend = []
        if has_pycuda:
            vbackend.append("pycuda")
        if has_cupy:
            vbackend.append("cupy")
        if has_pyopencl:
            vbackend.append("pyopencl")
        for backend in vbackend:
            vtype = (np.float32, np.float64)
            if backend == "pyopencl" and not has_cl_fp64:
                vtype = (np.float32,)
            for dtype in vtype:
                vlut = [None]
                if dtype == np.float32:
                    vlut += [True]
                for inplace in [True, False]:
                    for norm in [0, 1]:
                        for lut in vlut:
                            step = 2 if inplace else 1
                            self.run_exhaustive(backend, range(2, 15000), ndim=1, dtype=dtype, inplace=inplace,
                                                norm=norm, use_lut=lut, r2c=True, nproc=16, verbose=True)
                            self.run_exhaustive(backend, range(2, 4500, step), ndim=2, dtype=dtype, inplace=inplace,
                                                norm=norm, use_lut=lut, r2c=True, nproc=16, verbose=True)
                            self.run_exhaustive(backend, range(2, 550, step), ndim=3, dtype=dtype, inplace=inplace,
                                                norm=norm, use_lut=lut, r2c=True, nproc=4, verbose=True)

    def _test_systematic_dct(self):
        """Systematic DCT tests, without shuffling axes"""
        vbackend = []
        if has_pycuda:
            vbackend.append("pycuda")
        if has_cupy:
            vbackend.append("cupy")
        if has_pyopencl:
            vbackend.append("pyopencl")
        for backend in vbackend:
            vtype = (np.float32, np.float64)
            if backend == "pyopencl" and not has_cl_fp64:
                vtype = (np.float32,)
            for dct in range(1, 4 + 1):
                for dtype in vtype:
                    vlut = [None]
                    if dtype == np.float32:
                        vlut += [True]
                    for inplace in [True, False]:
                        for lut in vlut:
                            # Not sure what the allowed range is for DCT, so test a smaller one
                            self.run_exhaustive(backend, range(2, 550), ndim=1, dtype=dtype, inplace=inplace,
                                                norm=1, use_lut=lut, dct=dct, nproc=8, verbose=True)
                            self.run_exhaustive(backend, range(2, 550), ndim=2, dtype=dtype, inplace=inplace,
                                                norm=1, use_lut=lut, dct=dct, nproc=8, verbose=True)
                            self.run_exhaustive(backend, range(2, 275), ndim=3, dtype=dtype, inplace=inplace,
                                                norm=1, use_lut=lut, dct=dct, nproc=4, verbose=True)


def suite():
    test_suite = unittest.TestSuite()
    load_tests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(load_tests(TestFFT))
    if "--exhaustive" in sys.argv:
        test_suite.addTest(load_tests(TestFFTSystematic))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite', verbosity=2)
    # print(exhaustive_test("pycuda", range(2, 1100), ndim=1, dtype=np.float32, inplace=True, norm=0, use_lut=None))
    # print(exhaustive_test("pycuda", range(2, 4500), ndim=2, dtype=np.float32, inplace=True, norm=0, use_lut=None))
    # print(exhaustive_test("pycuda", range(2, 550), ndim=3, dtype=np.float32, inplace=True, norm=0, use_lut=None))
