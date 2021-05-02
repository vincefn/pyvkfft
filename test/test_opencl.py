import unittest
import os
import numpy as np
from numpy.fft import fftn, ifftn, fftshift, rfftn, irfftn

try:
    from scipy.misc import ascent
except ImportError:
    def ascent():
        return np.random.randint(0, 255, (512, 512))
from pyvkfft.opencl import VkFFTApp

try:
    import pyopencl as cl
    import pyopencl.array as cla
except ImportError:
    cla = None


class TestVkFFTOpenCL(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
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

    def test_pyopencl(self):
        self.assertTrue(cla is not None, "pyopencl is not available")

    @unittest.skipIf(cla is None, "pyopencl is not available")
    def test_c2c(self):
        """
        Test inplace C2C transforms
        """
        n = 32
        for dims in range(1, 5):
            if dims >=3:
                ndim_max = min(dims + 1, 2)
            else:
                ndim_max = min(dims + 1, 3)
            for ndim in range(1, ndim_max):
                for dtype in [np.complex64, np.complex128]:
                    for norm in [0, 1, "ortho"]:
                        with self.subTest(dims=dims, ndim=ndim, dtype=dtype, norm=norm):
                            if dtype == np.complex64:
                                rtol = 1e-6
                            else:
                                rtol = 1e-12

                            d = np.random.uniform(0, 1, [n] * dims).astype(dtype)
                            # A pure random array may not be a very good test (too random),
                            # so add a Gaussian
                            xx = [np.fft.fftshift(np.fft.fftfreq(n))] * dims
                            v = np.zeros_like(d)
                            for x in np.meshgrid(*xx, indexing='ij'):
                                v += x ** 2
                            d += 10 * np.exp(-v * 2)
                            n0 = (abs(d) ** 2).sum()
                            d_gpu = cla.to_device(self.queue, d)
                            app = VkFFTApp(d.shape, d.dtype, self.queue, ndim=ndim, norm=norm)
                            # base FFT scale
                            s = np.sqrt(np.prod(d.shape[-ndim:]))

                            d = fftn(d, axes=list(range(dims))[-ndim:]) / s
                            d_gpu = app.fft(d_gpu)
                            d_gpu *= dtype(app.get_fft_scale())
                            self.assertTrue(np.allclose(d, d_gpu.get(), rtol=rtol, atol=abs(d).max() * rtol))

                            d = ifftn(d, axes=list(range(dims))[-ndim:]) * s
                            app.ifft(d_gpu)
                            d_gpu *= dtype(app.get_ifft_scale())
                            self.assertTrue(np.allclose(d, d_gpu.get(), rtol=rtol, atol=abs(d).max() * rtol))
                            n1 = (abs(d_gpu.get()) ** 2).sum()
                            self.assertTrue(np.isclose(n0, n1, rtol=rtol))

    @unittest.skipIf(cla is None, "pyopencl is not available")
    def test_c2c_outofplace(self):
        """
        Test out-of-place C2C transforms
        """
        n = 32
        for dims in range(1, 5):
            if dims >=3:
                ndim_max = min(dims + 1, 2)
            else:
                ndim_max = min(dims + 1, 3)
            for ndim in range(1, ndim_max):
                for dtype in [np.complex64, np.complex128]:
                    for norm in [0, 1, "ortho"]:
                        with self.subTest(dims=dims, ndim=ndim, dtype=dtype, norm=norm):
                            if dtype == np.complex64:
                                rtol = 1e-6
                            else:
                                rtol = 1e-12

                            d = np.random.uniform(0, 1, [n] * dims).astype(dtype)
                            # A pure random array may not be a very good test (too random),
                            # so add a Gaussian
                            xx = [np.fft.fftshift(np.fft.fftfreq(n))] * dims
                            v = np.zeros_like(d)
                            for x in np.meshgrid(*xx, indexing='ij'):
                                v += x ** 2
                            d += 10 * np.exp(-v * 2)
                            n0 = (abs(d) ** 2).sum()
                            d_gpu = cla.to_device(self.queue, d)
                            d1_gpu = cla.zeros_like(d_gpu)
                            app = VkFFTApp(d.shape, d.dtype, self.queue, ndim=ndim, norm=norm, inplace=False)
                            # base FFT scale
                            s = np.sqrt(np.prod(d.shape[-ndim:]))

                            d = fftn(d, axes=list(range(dims))[-ndim:]) / s
                            app.fft(d_gpu, d1_gpu)
                            d1_gpu *= dtype(app.get_fft_scale())
                            self.assertTrue(np.allclose(d, d1_gpu.get(), rtol=rtol, atol=abs(d).max() * rtol))

                            d = ifftn(d, axes=list(range(dims))[-ndim:]) * s
                            app.ifft(d1_gpu, d_gpu)
                            d_gpu *= dtype(app.get_ifft_scale())
                            self.assertTrue(np.allclose(d, d_gpu.get(), rtol=rtol, atol=abs(d).max() * rtol))
                            n1 = (abs(d_gpu.get()) ** 2).sum()
                            self.assertTrue(np.isclose(n0, n1, rtol=rtol))

    @unittest.skipIf(cla is None, "pyopencl is not available")
    def test_r2c(self):
        """
        Test inplace R2C transforms
        """
        n = 32
        for dims in range(1, 5):
            if dims >=3:
                ndim_max = min(dims + 1, 2)
            else:
                ndim_max = min(dims + 1, 3)
            for ndim in range(1, ndim_max):
                for dtype in [np.float32, np.float64]:
                    for norm in [0, 1, "ortho"]:
                        with self.subTest(dims=dims, ndim=ndim, dtype=dtype, norm=norm):
                            if dtype == np.float32:
                                rtol = 1e-6
                                c_dtype = np.complex64
                            else:
                                rtol = 1e-12
                                c_dtype = np.complex128

                            sh = [n] * dims
                            sh[-1] += 2
                            shc = [n] * dims
                            shc[-1] = n // 2 + 1

                            d = np.random.uniform(0, 1, sh).astype(dtype)
                            # A pure random array may not be a very good test (too random),
                            # so add a Gaussian
                            xx = [np.fft.fftshift(np.fft.fftfreq(nn)) for nn in sh]
                            v = np.zeros_like(d)
                            for x in np.meshgrid(*xx, indexing='ij'):
                                v += x ** 2
                            d += 10 * np.exp(-v * 2)
                            n0 = (abs(d[..., :-2]) ** 2).sum()
                            d_gpu = cla.to_device(self.queue, d)
                            app = VkFFTApp(d.shape, d.dtype, self.queue, ndim=ndim, norm=norm, r2c=True)
                            # base FFT scale
                            s = np.sqrt(n ** ndim)

                            d = rfftn(d[..., :-2], axes=list(range(dims))[-ndim:]) / s
                            d_gpu = app.fft(d_gpu)
                            d_gpu *= c_dtype(app.get_fft_scale())
                            self.assertTrue(d_gpu.shape == tuple(shc))

                            if dtype == np.float32:
                                self.assertTrue(d_gpu.dtype == np.complex64)
                            elif dtype == np.float64:
                                self.assertTrue(d_gpu.dtype == np.complex128)

                            self.assertTrue(np.allclose(d, d_gpu.get(), rtol=rtol, atol=abs(d).max() * rtol))

                            d = irfftn(d, axes=list(range(dims))[-ndim:]) * s
                            d_gpu = app.ifft(d_gpu)
                            d_gpu *= dtype(app.get_ifft_scale())
                            self.assertTrue(d_gpu.shape == tuple(sh))

                            self.assertTrue(np.allclose(d, d_gpu.get()[..., :-2], rtol=rtol, atol=abs(d).max() * rtol))
                            n1 = (abs(d_gpu.get()[..., :-2]) ** 2).sum()
                            self.assertTrue(np.isclose(n0, n1, rtol=rtol))

    @unittest.skipIf(cla is None, "pyopencl is not available")
    def test_r2c_outofplace(self):
        """
        Test out-of-place R2C transforms
        """
        n = 32
        for dims in range(1, 5):
            if dims >=3:
                ndim_max = min(dims + 1, 2)
            else:
                ndim_max = min(dims + 1, 3)
            for ndim in range(1, ndim_max):
                for dtype in [np.float32, np.float64]:
                    for norm in [0, 1, "ortho"]:
                        with self.subTest(dims=dims, ndim=ndim, dtype=dtype, norm=norm):
                            if dtype == np.float32:
                                rtol = 1e-6
                            else:
                                rtol = 1e-12
                            if dtype == np.float32:
                                dtype_c = np.complex64
                            elif dtype == np.float64:
                                dtype_c = np.complex128

                            sh = [n] * dims
                            sh = tuple(sh)
                            shc = [n] * dims
                            shc[-1] = n // 2 + 1
                            shc = tuple(shc)

                            d = np.random.uniform(0, 1, sh).astype(dtype)
                            # A pure random array may not be a very good test (too random),
                            # so add a Gaussian
                            xx = [np.fft.fftshift(np.fft.fftfreq(nn)) for nn in sh]
                            v = np.zeros_like(d)
                            for x in np.meshgrid(*xx, indexing='ij'):
                                v += x ** 2
                            d += 10 * np.exp(-v * 2)
                            n0 = (abs(d) ** 2).sum()
                            d_gpu = cla.to_device(self.queue, d)
                            d1_gpu = cla.empty(self.queue, shc, dtype=dtype_c)

                            app = VkFFTApp(d.shape, d.dtype, self.queue, ndim=ndim, norm=norm, r2c=True, inplace=False)
                            # base FFT scale
                            s = np.sqrt(np.prod(d.shape[-ndim:]))

                            d = rfftn(d, axes=list(range(dims))[-ndim:]) / s
                            d1_gpu = app.fft(d_gpu, d1_gpu)
                            d1_gpu *= dtype(app.get_fft_scale())
                            self.assertTrue(d1_gpu.shape == tuple(shc))
                            self.assertTrue(d1_gpu.dtype == dtype_c)

                            self.assertTrue(np.allclose(d, d1_gpu.get(), rtol=rtol, atol=abs(d).max() * rtol))

                            d = irfftn(d, axes=list(range(dims))[-ndim:]) * s
                            d_gpu = app.ifft(d1_gpu, d_gpu)
                            d_gpu *= dtype(app.get_ifft_scale())
                            self.assertTrue(d_gpu.shape == tuple(sh))

                            self.assertTrue(np.allclose(d, d_gpu.get(), rtol=rtol, atol=abs(d).max() * rtol))
                            n1 = (abs(d_gpu.get()) ** 2).sum()
                            self.assertTrue(np.isclose(n0, n1, rtol=rtol))


def suite():
    test_suite = unittest.TestSuite()
    load_tests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(load_tests(TestVkFFTOpenCL))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
