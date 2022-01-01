import os
import unittest
import numpy as np

try:
    from scipy.misc import ascent
except ImportError:
    def ascent():
        return np.random.randint(0, 255, (512, 512))

from pyvkfft.fft import fftn as vkfftn, ifftn as vkifftn, rfftn as vkrfftn, \
    irfftn as vkirfftn, dctn as vkdctn, idctn as vkidctn
from pyvkfft.accuracy import test_accuracy, fftn, ifftn

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

    @classmethod
    def setUpClass(cls) -> None:
        cls.verbose = True
        if has_pyopencl:
            # Create some context on the first available GPU
            if 'PYOPENCL_CTX' in os.environ:
                cls.ctx = cl.create_some_context()
            else:
                cls.ctx = None
                # Find the first OpenCL GPU available and use it, unless
                for p in cl.get_platforms():
                    for d in p.get_devices():
                        if d.type & cl.device_type.GPU == 0:
                            continue
                        cls.ctx = cl.Context(devices=(d,))
                        break
                    if cls.ctx is not None:
                        break
            cls.queue = cl.CommandQueue(cls.ctx)
            if 'cl_khr_fp64' in cls.queue.device.extensions:
                cls.has_cl_fp64 = True
            else:
                cls.has_cl_fp64 = False

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
                dc = cla.to_device(self.queue, ascent().astype(np.complex64))
                dr = cla.to_device(self.queue, ascent().astype(np.float32))
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

    def run_fft(self, vbackend, vn, dims_max=4, ndim_max=3, vtype=(np.complex64, np.complex128),
                vlut="auto", vinplace=(True, False), vnorm=(0, 1),
                vr2c=(False,), vdct=(False,), verbose=False, dry_run=False):
        """
        Run a series of tests
        :param vbackend: list of backends to test among "pycuda", "cupy and "pyopencl"
        :param vn: list of transform sizes to test
        :param dims_max: max number of dimensions for the array (up to 4)
        :param ndim_max: max transform dimension
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
                                if not r2c and not dct:
                                    # Test custom axes only for C2C
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
                                                            n2, ni, n2i, nii, tol, dt1, dt2, dt3, dt4, \
                                                            src1, src2, res = \
                                                                test_accuracy(backend, sh, ndim, axes, dtype, inplace,
                                                                              norm, use_lut, r2c=r2c, dct=dct,
                                                                              stream=None, queue=self.queue,
                                                                              return_array=False, init_array=d0,
                                                                              verbose=verbose)
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
                if backend == "pyopencl" and not self.has_cl_fp64:
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
                if backend == "pyopencl" and not self.has_cl_fp64:
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
                if backend == "pyopencl" and not self.has_cl_fp64:
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


def suite():
    test_suite = unittest.TestSuite()
    load_tests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(load_tests(TestFFT))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite', verbosity=2)
