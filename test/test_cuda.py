import unittest
import numpy as np
from numpy.fft import fftn, ifftn, fftshift, rfftn, irfftn
from scipy.misc import ascent
from pyvkfft.cuda import VkFFTApp

try:
    import pycuda.autoinit
    import pycuda.driver as cu_drv
    import pycuda.gpuarray as cua
except:
    cua = None


class TestVkFFTCUDA(unittest.TestCase):

    def test_pycuda(self):
        self.assertTrue(cua is not None, "pycuda is not available")

    def test_c2c_inplace_2d(self):
        """
        Test inplace C2C transforms
        :return:
        """
        for dtype in [np.complex64, np.complex128]:
            if dtype == np.complex64:
                rtol = 1e-6
            else:
                rtol = 1e-12

            d = ascent().astype(dtype)
            n0 = (abs(d) ** 2).sum()
            d_cu = cua.to_gpu(d)
            app = VkFFTApp(d.shape, d.dtype, ndim=2, norm=1)

            d = fftn(d)
            app.fft(d_cu)
            self.assertTrue(np.allclose(d, d_cu.get(), rtol=rtol, atol=abs(d).max() * rtol))

            d = ifftn(d)
            app.ifft(d_cu)
            self.assertTrue(np.allclose(d, d_cu.get(), rtol=rtol, atol=abs(d).max() * rtol))
            n1 = (abs(d_cu.get()) ** 2).sum()
            self.assertTrue(np.isclose(n0, n1, rtol=rtol))

    def test_c2c_inplace_2d_n(self):
        """
        Test inplace C2C 2D FFT transforms on a 3D array.
        :return:
        """
        for dtype in [np.complex64, np.complex128]:
            if dtype == np.complex64:
                rtol = 1e-6
            else:
                rtol = 1e-12
            a = ascent().astype(dtype)
            ny, nx = a.shape
            d = np.empty((17, ny, nx), dtype=dtype)
            for i in range(len(d)):
                d[i] = np.roll(a, np.random.randint(0, 100, 2)) * \
                       np.exp(0.01j * np.roll(a, np.random.randint(0, 100, 2)))
            n0 = (abs(d) ** 2).sum()
            d_cu = cua.to_gpu(d)
            app = VkFFTApp(d.shape, d.dtype, ndim=2, norm=1)

            d = fftn(d, axes=(1, 2))
            app.fft(d_cu)
            self.assertTrue(np.allclose(d, d_cu.get(), rtol=rtol, atol=abs(d).max() * rtol))

            d = ifftn(d, axes=(1, 2))
            app.ifft(d_cu)
            self.assertTrue(np.allclose(d, d_cu.get(), rtol=rtol, atol=abs(d).max() * rtol))
            n1 = (abs(d_cu.get()) ** 2).sum()
            self.assertTrue(np.isclose(n0, n1, rtol=rtol))

    def test_c2c_inplace_3d(self):
        """
        Test inplace C2C 3D transforms
        :return:
        """
        for dtype in [np.complex64, np.complex128]:
            if dtype == np.complex64:
                rtol = 1e-6
            else:
                rtol = 1e-12

            a = ascent().astype(dtype)
            ny, nx = a.shape
            d = np.empty((3 * 13, ny, nx), dtype=dtype)
            for i in range(len(d)):
                d[i] = np.roll(a, np.random.randint(0, 100, 2)) * \
                       np.exp(0.01j * np.roll(a, np.random.randint(0, 100, 2)))
            n0 = (abs(d) ** 2).sum()
            d_cu = cua.to_gpu(d)
            app = VkFFTApp(d.shape, d.dtype, ndim=3, norm=1)

            d = fftn(d)
            app.fft(d_cu)
            self.assertTrue(np.allclose(d, d_cu.get(), rtol=rtol, atol=abs(d).max() * rtol))

            d = ifftn(d)
            app.ifft(d_cu)
            self.assertTrue(np.allclose(d, d_cu.get(), rtol=rtol, atol=abs(d).max() * rtol))
            n1 = (abs(d_cu.get()) ** 2).sum()
            self.assertTrue(np.isclose(n0, n1, rtol=rtol))

    def test_c2c_outofplace_2d(self):
        """
        Test out-of-place C2C transforms
        :return:
        """
        for dtype in [np.complex64, np.complex128]:
            if dtype == np.complex64:
                rtol = 1e-6
            else:
                rtol = 1e-12

            d1 = ascent().astype(dtype)
            n0 = (abs(d1) ** 2).sum()
            d1_cu = cua.to_gpu(d1)
            d2_cu = cua.empty_like(d1_cu)
            app = VkFFTApp(d1.shape, d1.dtype, ndim=2, norm=1, inplace=False)

            d2 = fftn(d1)
            app.fft(d1_cu, d2_cu)
            # Check original array is unchanged and compare result with numpy.fft
            self.assertTrue(np.allclose(d1, d1_cu.get(), rtol=rtol, atol=abs(d1).max() * rtol))
            self.assertTrue(np.allclose(d2, d2_cu.get(), rtol=rtol, atol=abs(d2).max() * rtol))

            d1 = ifftn(d2)
            app.ifft(d2_cu, d1_cu)
            self.assertTrue(np.allclose(d1, d1_cu.get(), rtol=rtol, atol=abs(d1).max() * rtol))
            self.assertTrue(np.allclose(d2, d2_cu.get(), rtol=rtol, atol=abs(d2).max() * rtol))
            n1 = (abs(d1_cu.get()) ** 2).sum()
            self.assertTrue(np.isclose(n0, n1, rtol=rtol))

    def test_c2c_outofplace_2d_n(self):
        """
        Test out-of-place C2C 2D FFT transforms on a 3D array
        :return:
        """
        for dtype in [np.complex64, np.complex128]:
            if dtype == np.complex64:
                rtol = 1e-6
            else:
                rtol = 1e-12
            a = ascent().astype(dtype)
            ny, nx = a.shape
            d1 = np.empty((17, ny, nx), dtype=dtype)
            for i in range(len(d1)):
                d1[i] = np.roll(a, np.random.randint(0, 100, 2)) * \
                        np.exp(0.01j * np.roll(a, np.random.randint(0, 100, 2)))

            n0 = (abs(d1) ** 2).sum()
            d1_cu = cua.to_gpu(d1)
            d2_cu = cua.empty_like(d1_cu)
            app = VkFFTApp(d1.shape, d1.dtype, ndim=2, norm=1, inplace=False)

            d2 = fftn(d1, axes=(1, 2))
            app.fft(d1_cu, d2_cu)
            # Check original array is unchanged and compare result with numpy.fft
            self.assertTrue(np.allclose(d1, d1_cu.get(), rtol=rtol, atol=abs(d1).max() * rtol))
            self.assertTrue(np.allclose(d2, d2_cu.get(), rtol=rtol, atol=abs(d2).max() * rtol))

            d1 = ifftn(d2, axes=(1, 2))
            app.ifft(d2_cu, d1_cu)
            self.assertTrue(np.allclose(d1, d1_cu.get(), rtol=rtol, atol=abs(d1).max() * rtol))
            self.assertTrue(np.allclose(d2, d2_cu.get(), rtol=rtol, atol=abs(d2).max() * rtol))
            n1 = (abs(d1_cu.get()) ** 2).sum()
            self.assertTrue(np.isclose(n0, n1, rtol=rtol))

    def test_c2c_outofplace_3d(self):
        """
        Test out-of-place C2C 3D FFT transforms
        :return:
        """
        for dtype in [np.complex64, np.complex128]:
            if dtype == np.complex64:
                rtol = 1e-6
            else:
                rtol = 1e-12
            a = ascent().astype(dtype)
            ny, nx = a.shape
            d1 = np.empty((3 * 13, ny, nx), dtype=dtype)
            for i in range(len(d1)):
                d1[i] = np.roll(a, np.random.randint(0, 100, 2)) * \
                        np.exp(0.01j * np.roll(a, np.random.randint(0, 100, 2)))

            n0 = (abs(d1) ** 2).sum()
            d1_cu = cua.to_gpu(d1)
            d2_cu = cua.empty_like(d1_cu)
            app = VkFFTApp(d1.shape, d1.dtype, ndim=3, norm=1, inplace=False)

            d2 = fftn(d1)
            app.fft(d1_cu, d2_cu)
            # Check original array is unchanged and compare result with numpy.fft
            self.assertTrue(np.allclose(d1, d1_cu.get(), rtol=rtol, atol=abs(d1).max() * rtol))
            self.assertTrue(np.allclose(d2, d2_cu.get(), rtol=rtol, atol=abs(d2).max() * rtol))

            d1 = ifftn(d2)
            app.ifft(d2_cu, d1_cu)
            self.assertTrue(np.allclose(d1, d1_cu.get(), rtol=rtol, atol=abs(d1).max() * rtol))
            self.assertTrue(np.allclose(d2, d2_cu.get(), rtol=rtol, atol=abs(d2).max() * rtol))
            n1 = (abs(d1_cu.get()) ** 2).sum()
            self.assertTrue(np.isclose(n0, n1, rtol=rtol))

    def test_R2C_C2R_inplace_2d(self):
        """
        Test real <-> complex (half-hermitian) for a couple of R2C and C2R transforms
        :return:
        """
        for dtype in [np.float32, np.float64]:
            if dtype == np.float32:
                rtol = 1e-6
            else:
                rtol = 1e-12

            d = np.zeros((512, 514), dtype=dtype)
            d[:, :512] = ascent()
            ny, nx = d.shape
            d_cu = cua.to_gpu(d)
            app = VkFFTApp(d.shape, d.dtype, ndim=2, r2c=True)
            d_cu = app.fft(d_cu)

            if dtype == np.float32:
                self.assertTrue(d_cu.dtype == np.complex64)
            elif dtype == np.float64:
                self.assertTrue(d_cu.dtype == np.complex128)
            self.assertTrue(d_cu.shape == (ny, (nx - 2) // 2 + 1))

            dn = rfftn(d[:, :-2])
            self.assertTrue(np.allclose(d_cu.get(), dn, rtol=rtol, atol=dn.max() * rtol),
                            "Compare VkFFT R2C transform with numpy rfftn")

            d_cu = app.ifft(d_cu)
            self.assertTrue(np.allclose(d_cu.get()[:, :-2], d[:, :-2], rtol=rtol, atol=d.max() * rtol),
                            "Compare VkFFT R2C+C2R transform")

    def test_R2C_C2R_inplace_2d_n(self):
        """
        Test real <-> complex (half-hermitian) 2D FFT for a couple of R2C and C2R transforms,
        in a 3D array
        :return:
        """
        for dtype in [np.float32, np.float64]:
            if dtype == np.float32:
                rtol = 1e-6
            else:
                rtol = 1e-12

            d = np.zeros((17, 512, 514), dtype=dtype)
            for i in range(len(d)):
                d[i, :, :512] = ascent()
            ny, nx = d.shape[-2:]
            d_cu = cua.to_gpu(d)
            app = VkFFTApp(d.shape, d.dtype, ndim=2, r2c=True)
            d_cu = app.fft(d_cu)

            if dtype == np.float32:
                self.assertTrue(d_cu.dtype == np.complex64)
            elif dtype == np.float64:
                self.assertTrue(d_cu.dtype == np.complex128)
            self.assertTrue(d_cu.shape == (len(d), ny, (nx - 2) // 2 + 1))

            dn = rfftn(d[..., :-2], axes=(1, 2))
            self.assertTrue(np.allclose(d_cu.get(), dn, rtol=rtol, atol=dn.max() * rtol),
                            "Compare VkFFT R2C transform with numpy rfftn")

            d_cu = app.ifft(d_cu)
            self.assertTrue(np.allclose(d_cu.get()[..., :-2], d[..., :-2], rtol=rtol, atol=d.max() * rtol),
                            "Compare VkFFT R2C+C2R transform")

    def test_R2C_C2R_inplace_3d(self):
        """
        Test real <-> complex (half-hermitian) 3D FFT for a couple of R2C and C2R transforms
        :return:
        """
        for dtype in [np.float32, np.float64]:
            if dtype == np.float32:
                rtol = 1e-6
            else:
                rtol = 1e-12

            d = np.zeros((3 * 13, 512, 514), dtype=dtype)
            for i in range(len(d)):
                d[i, :, :512] = ascent()
            ny, nx = d.shape[-2:]
            d_cu = cua.to_gpu(d)
            app = VkFFTApp(d.shape, d.dtype, ndim=3, r2c=True)
            d_cu = app.fft(d_cu)

            if dtype == np.float32:
                self.assertTrue(d_cu.dtype == np.complex64)
            elif dtype == np.float64:
                self.assertTrue(d_cu.dtype == np.complex128)
            self.assertTrue(d_cu.shape == (len(d), ny, (nx - 2) // 2 + 1))

            dn = rfftn(d[..., :-2])
            self.assertTrue(np.allclose(d_cu.get(), dn, rtol=rtol, atol=dn.max() * rtol),
                            "Compare VkFFT R2C 3D transform with numpy rfftn")

            d_cu = app.ifft(d_cu)
            self.assertTrue(np.allclose(d_cu.get()[..., :-2], d[..., :-2], rtol=rtol, atol=d.max() * rtol),
                            "Compare VkFFT R2C+C2R 3D transform")

    def test_streams(self):
        """
        Test multiple FFT in // with different streams.
        :return:
        """
        for dtype in [np.complex64, np.complex128]:
            if dtype == np.complex64:
                rtol = 1e-6
            else:
                rtol = 1e-12
            d = ascent().astype(dtype)
            n_streams = 5
            vd = []
            vapp = []
            for i in range(n_streams):
                vd.append(cua.to_gpu(np.roll(d, i * 7, axis=1)))
                vapp.append(VkFFTApp(d.shape, d.dtype, ndim=2, norm=1, stream=cu_drv.Stream()))

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
    test_suite.addTest(load_tests(TestVkFFTCUDA))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
