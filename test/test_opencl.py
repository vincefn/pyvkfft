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
                    if d.type & cl.device_type.GPU is False:
                        continue
                    cls.ctx = cl.Context(devices=(d,))
                    break
                if cls.ctx is not None:
                    break
        cls.queue = cl.CommandQueue(cls.ctx)

    def test_pyopencl(self):
        self.assertTrue(cla is not None, "pyopencl is not available")

    def test_c2c(self):
        """
        Test inplace C2C transform
        """
        n = 64
        for dims in range(1, 4):
            for ndim in range(1, dims + 1):
                for dtype in [np.complex64, np.complex128]:
                    with self.subTest(dims=dims, ndim=ndim, dtype=dtype):
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
                        d_gpu = cla.to_device(self.queue,d)
                        app = VkFFTApp(d.shape, d.dtype, self.queue, ndim=ndim, norm=1)
                        # base FFT scale
                        s = np.sqrt(np.prod(d.shape[-ndim:]))

                        d = fftn(d, axes=list(range(dims))[-ndim:]) / s
                        app.fft(d_gpu)
                        print("%12f %12f %12f" % (n0, (abs(d)**2).sum(), (abs(d_gpu.get())**2).sum()), dims, ndim, dtype)
                        self.assertTrue(np.allclose(d, d_gpu.get(), rtol=rtol, atol=rtol))

                        d = ifftn(d)
                        app.ifft(d_gpu)
                        print("%12f %12f %12f" % (n0, (abs(d)**2).sum(), (abs(d_gpu.get())**2).sum()), dims, ndim, dtype)
                        self.assertTrue(np.allclose(d, d_gpu.get(), rtol=rtol, atol=rtol))
                        n1 = (abs(d_gpu.get()) ** 2).sum()
                        self.assertTrue(np.isclose(n0, n1, rtol=rtol))


def suite():
    test_suite = unittest.TestSuite()
    load_tests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(load_tests(TestVkFFTOpenCL))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
