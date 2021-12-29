# -*- coding: utf-8 -*-

# PyVkFFT
#   (c) 2021- : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr
#
#
# Unit tests for the (py)opencl backend.
# WARNING: these take a LONG time. There are few tests, but each has many sub-tests
# for the different backends, dimensions, transform type, accuracy and normalisation
# For example the C2C currently has 216 sub-tests (108 if double precision is not available)...

import unittest
import os
import numpy as np
from numpy.fft import fftn, ifftn, fftshift, rfftn, irfftn

try:
    from scipy.fft import dctn, idctn

    has_scipy_dct = True
except ImportError:
    has_scipy_dct = False
    print("Install scipy if you want to test dct transforms")

try:
    from scipy.misc import ascent
except ImportError:
    def ascent():
        return np.random.randint(0, 255, (512, 512))

try:
    import pyopencl as cl
    import pyopencl.array as cla
    from pyvkfft.opencl import VkFFTApp, primes
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
        cls.dtype_float_v = [np.float32]
        cls.dtype_complex_v = [np.complex64]
        if 'cl_khr_fp64' in cls.queue.device.extensions:
            cls.dtype_float_v.append(np.float64)
            cls.dtype_complex_v.append(np.complex128)
        cls.vn = [30, 34, 808]

    def test_pyopencl(self):
        """Test if pyopencl available (may not matter if pycuda or cupy is used instead)"""
        self.assertTrue(cla is not None, "pyopencl is not available")

    @unittest.skipIf(cla is None, "pyopencl is not available")
    def test_c2c(self):
        """
        Test inplace C2C transforms
        """
        for n in self.vn:  # test both radix-2 and Bluestein algorithms
            max_dim = 4
            if n > 64:
                # Only test 1D and 2D for large sizes
                max_dim = 2
            for dims in range(1, max_dim + 1):
                for ndim0 in range(1, min(dims, 3) + 1):
                    # Setup use of either ndim or axes, also test skipping dimensions
                    ndim_axes = [(ndim0, None)]
                    for i in range(1, 2 ** (ndim0 - 1)):
                        axes = []
                        for ii in range(ndim0):
                            if not (i & 2 ** ii):
                                axes.append(-ii - 1)
                        ndim_axes.append((None, axes))
                    for ndim, axes in ndim_axes:
                        sh = [n] * dims
                        if axes is not None:
                            for i in range(dims):
                                if -1 - i not in axes:
                                    # Smaller array for non-transformed axes
                                    sh[-1 - i] = 2
                        else:
                            for i in range(ndim, dims):
                                sh[-1 - i] = 2

                        d0 = np.random.uniform(0, 1, sh)
                        # A pure random array may not be a very good test (too random),
                        # so add a Gaussian
                        xx = [np.fft.fftshift(np.fft.fftfreq(nx)) for nx in sh]
                        v = np.zeros_like(d0)
                        for x in np.meshgrid(*xx, indexing='ij'):
                            v += x ** 2
                        d0 += 10 * np.exp(-v * 2)
                        for dtype in self.dtype_complex_v:
                            for norm in [0, 1, "ortho"]:
                                with self.subTest(n=n, dims=dims, ndim=ndim, axes=axes, dtype=dtype, norm=norm):
                                    if dtype == np.complex64:
                                        rtol = 1e-4
                                    else:
                                        rtol = 1e-8
                                    if max(primes(n)) > 13:
                                        # Lower accuracy for Bluestein algorithm.
                                        rtol *= 4
                                        if n > 100:
                                            rtol *= 4
                                            if dtype == np.complex128:
                                                rtol *= 4

                                    d = d0.astype(dtype).copy()
                                    n0 = (abs(d) ** 2).sum()
                                    d_gpu = cla.to_device(self.queue, d)
                                    app = VkFFTApp(d.shape, d.dtype, self.queue, ndim=ndim, norm=norm, axes=axes)
                                    if axes is None:
                                        axes = list(range(dims))[-ndim:]  # For numpy
                                    # base FFT scale for numpy
                                    s = np.sqrt(np.prod([d.shape[i] for i in axes]))

                                    d = fftn(d, axes=axes) / s
                                    d_gpu = app.fft(d_gpu) * app.get_fft_scale()
                                    self.assertTrue(np.allclose(d, d_gpu.get(), rtol=rtol, atol=abs(d).max() * rtol))

                                    d = ifftn(d, axes=axes) * s
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
        for n in self.vn:  # test both radix-2 and Bluestein algorithms
            max_dim = 4
            if n > 64:
                # Only test 1D and 2D for large sizes
                max_dim = 2
            for dims in range(1, max_dim + 1):
                for ndim0 in range(1, min(dims, 3) + 1):
                    # Setup use of either ndim or axes, also test skipping dimensions
                    ndim_axes = [(ndim0, None)]
                    for i in range(1, 2 ** (ndim0 - 1)):
                        axes = []
                        for ii in range(ndim0):
                            if not (i & 2 ** ii):
                                axes.append(-ii - 1)
                        ndim_axes.append((None, axes))
                    for ndim, axes in ndim_axes:
                        sh = [n] * dims
                        if axes is not None:
                            for i in range(dims):
                                if -1 - i not in axes:
                                    # Smaller array for non-transformed axes
                                    sh[-1 - i] = 2
                        else:
                            for i in range(ndim, dims):
                                sh[-1 - i] = 2

                        d0 = np.random.uniform(0, 1, sh)
                        # A pure random array may not be a very good test (too random),
                        # so add a Gaussian
                        xx = [np.fft.fftshift(np.fft.fftfreq(nx)) for nx in sh]
                        v = np.zeros_like(d0)
                        for x in np.meshgrid(*xx, indexing='ij'):
                            v += x ** 2
                        d0 += 10 * np.exp(-v * 2)
                        for dtype in self.dtype_complex_v:
                            for norm in [0, 1, "ortho"]:
                                with self.subTest(n=n, dims=dims, ndim=ndim, axes=axes, dtype=dtype, norm=norm):
                                    if dtype == np.complex64:
                                        rtol = 1e-4
                                    else:
                                        rtol = 1e-8
                                    if max(primes(n)) > 13:
                                        # Lower accuracy for Bluestein algorithm.
                                        rtol *= 4
                                        if n > 100:
                                            rtol *= 4
                                            if dtype == np.complex128:
                                                rtol *= 4

                                    d = d0.astype(dtype).copy()
                                    n0 = (abs(d) ** 2).sum()
                                    d_gpu = cla.to_device(self.queue, d)
                                    d1_gpu = cla.zeros_like(d_gpu)
                                    app = VkFFTApp(d.shape, d.dtype, self.queue, ndim=ndim, norm=norm,
                                                   axes=axes, inplace=False)
                                    if axes is None:
                                        axes = list(range(dims))[-ndim:]  # For numpy
                                    # base FFT scale for numpy
                                    s = np.sqrt(np.prod([d.shape[i] for i in axes]))

                                    d = fftn(d, axes=axes) / s
                                    app.fft(d_gpu, d1_gpu)
                                    d1_gpu *= app.get_fft_scale()
                                    self.assertTrue(np.allclose(d, d1_gpu.get(), rtol=rtol, atol=abs(d).max() * rtol))

                                    d = ifftn(d, axes=axes) * s
                                    app.ifft(d1_gpu, d_gpu)
                                    d_gpu *= app.get_ifft_scale()
                                    n1 = (abs(d_gpu.get()) ** 2).sum()
                                    self.assertTrue(np.allclose(d, d_gpu.get(), rtol=rtol, atol=abs(d).max() * rtol))
                                    self.assertTrue(np.isclose(n0, n1, rtol=rtol))

    @unittest.skipIf(cla is None, "pyopencl is not available")
    def test_r2c(self):
        """
        Test inplace R2C transforms
        """
        for n in self.vn:
            # test both radix-2 and Bluestein algorithms.
            # We need a multiple of 2 for the first axis of an inplace R2C transform
            if n % 2:
                continue
            max_dim = 4
            if n > 64:
                # Only test 1D and 2D for large sizes
                max_dim = 2
            for dims in range(1, max_dim + 1):
                for ndim in range(1, min(dims, 3) + 1):
                    sh = [n] * dims
                    sh[-1] += 2
                    shc = [n] * dims
                    shc[-1] = n // 2 + 1
                    for i in range(ndim, dims):
                        # Smaller array on non-transformed axes
                        sh[-i - 1] = 2
                        shc[-i - 1] = 2

                    d0 = np.random.uniform(0, 1, sh)
                    # A pure random array may not be a very good test (too random),
                    # so add a Gaussian
                    xx = [np.fft.fftshift(np.fft.fftfreq(nn)) for nn in sh]
                    v = np.zeros_like(d0)
                    for x in np.meshgrid(*xx, indexing='ij'):
                        v += x ** 2
                    d0 += 10 * np.exp(-v * 2)
                    for dtype in self.dtype_float_v:
                        for norm in [0, 1, "ortho"]:
                            with self.subTest(n=n, dims=dims, ndim=ndim, dtype=dtype, norm=norm):
                                if dtype == np.float32:
                                    rtol = 1e-4
                                    c_dtype = np.complex64
                                else:
                                    rtol = 1e-8
                                    c_dtype = np.complex128
                                if max(primes(n)) > 13:
                                    rtol *= 4  # Lower accuracy for Bluestein algorithm
                                    if n > 100 and dtype == np.float64:
                                        rtol *= 10

                                d = d0.astype(dtype).copy()
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
                                d_gpu = app.ifft(d_gpu) * app.get_ifft_scale()
                                self.assertTrue(d_gpu.shape == tuple(sh))

                                n1 = (abs(d_gpu.get()[..., :-2]) ** 2).sum()
                                # print(dims, ndim, n, "%6s" % str(norm), "%14.1f %14.1f" % (n0, n1),
                                #       np.isclose(n0, n1, rtol=rtol))
                                self.assertTrue(np.isclose(n0, n1, rtol=rtol))
                                self.assertTrue(
                                    np.allclose(d, d_gpu.get()[..., :-2], rtol=rtol, atol=abs(d).max() * rtol))

    @unittest.skipIf(cla is None, "pyopencl is not available")
    def test_r2c_outofplace(self):
        """
        Test out-of-place R2C transforms
        """
        for n in self.vn:  # test both radix-2 and Bluestein algorithms.
            max_dim = 4
            if n > 64:
                # Only test 1D and 2D for large sizes
                max_dim = 2
            for dims in range(1, max_dim + 1):
                for ndim in range(1, min(dims, 3) + 1):
                    sh = [n] * dims
                    shc = [n] * dims
                    shc[-1] = n // 2 + 1
                    for i in range(ndim, dims):
                        # Smaller array on non-transformed axes
                        sh[-i - 1] = 2
                        shc[-i - 1] = 2
                    sh = tuple(sh)
                    shc = tuple(shc)

                    d0 = np.random.uniform(0, 1, sh)
                    # A pure random array may not be a very good test (too random),
                    # so add a Gaussian
                    xx = [np.fft.fftshift(np.fft.fftfreq(nn)) for nn in sh]
                    v = np.zeros_like(d0)
                    for x in np.meshgrid(*xx, indexing='ij'):
                        v += x ** 2
                    d0 += 10 * np.exp(-v * 2)
                    for dtype in self.dtype_float_v:
                        for norm in [0, 1, "ortho"]:
                            with self.subTest(n=n, dims=dims, ndim=ndim, dtype=dtype, norm=norm):
                                if dtype == np.float32:
                                    rtol = 1e-4
                                else:
                                    rtol = 1e-8
                                if max(primes(n)) > 13:
                                    rtol *= 4  # Lower accuracy for Bluestein algorithm
                                    if n > 100 and dtype == np.float64:
                                        rtol *= 10

                                if dtype == np.float32:
                                    dtype_c = np.complex64
                                elif dtype == np.float64:
                                    dtype_c = np.complex128
                                d = d0.astype(dtype).copy()
                                n0 = (abs(d) ** 2).sum()
                                d_gpu = cla.to_device(self.queue, d)
                                d1_gpu = cla.empty(self.queue, shc, dtype=dtype_c)

                                app = VkFFTApp(d.shape, d.dtype, self.queue, ndim=ndim, norm=norm, r2c=True,
                                               inplace=False)
                                # base FFT scale
                                s = np.sqrt(np.prod(d.shape[-ndim:]))

                                d = rfftn(d, axes=list(range(dims))[-ndim:]) / s
                                d1_gpu = app.fft(d_gpu, d1_gpu) * app.get_fft_scale()
                                self.assertTrue(d1_gpu.shape == tuple(shc))
                                self.assertTrue(d1_gpu.dtype == dtype_c)

                                self.assertTrue(np.allclose(d, d1_gpu.get(), rtol=rtol, atol=abs(d).max() * rtol))

                                # The shape of the transformed axes must be supplied for scipy.fft.irfftn
                                axes = list(range(dims))[-ndim:]
                                d = irfftn(d, [sh[i] for i in axes], axes=axes) * s
                                d_gpu = app.ifft(d1_gpu, d_gpu) * app.get_ifft_scale()
                                self.assertTrue(d_gpu.shape == tuple(sh))

                                self.assertTrue(np.allclose(d, d_gpu.get(), rtol=rtol, atol=abs(d).max() * rtol))
                                n1 = (abs(d_gpu.get()) ** 2).sum()
                                self.assertTrue(np.isclose(n0, n1, rtol=rtol))

    @unittest.skipIf(cla is None, "pyopencl is not available")
    @unittest.skipIf(not has_scipy_dct, "scipy.fft.dctn is not available")
    def test_dct(self):
        """
        Test inplace DCT transforms
        """
        for n in self.vn:
            # test both radix-2 and Bluestein algorithms.
            max_dim = 4
            if n > 64:
                # Only test 1D and 2D for large sizes
                max_dim = 2
            for dims in range(1, max_dim + 1):
                for ndim in range(1, min(dims, 3) + 1):
                    sh = [n] * dims
                    for i in range(ndim, dims):
                        # Smaller array on non-transformed axes
                        sh[-i - 1] = 2
                    d0 = np.random.uniform(0, 1, sh)
                    # A pure random array may not be a very good test (too random),
                    # so add a Gaussian
                    xx = [np.fft.fftshift(np.fft.fftfreq(nn)) for nn in sh]
                    v = np.zeros_like(d0)
                    for x in np.meshgrid(*xx, indexing='ij'):
                        v += x ** 2
                    d0 += 10 * np.exp(-v * 2)
                    for dtype in self.dtype_float_v:
                        for dct in range(1, 4 + 1):
                            if dct == 1 and n > 512 * 4 // np.dtype(dtype).itemsize:
                                # Not sure where this limitation comes from
                                # limit is n > 513 for float64, 1025 for float32 on a GTX 1080
                                continue
                            elif n > 2000 * 4 // np.dtype(dtype).itemsize:
                                # limit is n > 2002 for float64, 4004 for float32 on a GTX 1080
                                continue
                            # for norm in [0, 1, "ortho"] # TODO : test all norms
                            norm = 1  # Same default norm as scipy
                            with self.subTest(n=n, dims=dims, ndim=ndim, dtype=dtype, dct=dct, norm=norm):
                                if dtype == np.float32:
                                    rtol = 1e-4
                                else:
                                    rtol = 1e-8
                                if max(primes(n)) > 13 or dct != 2:
                                    # Lower accuracy for Bluestein transforms
                                    # Why is it necessary for DCT types != 2 ?
                                    rtol *= 8

                                d = d0.astype(dtype).copy()

                                d_gpu = cla.to_device(self.queue, d)
                                app = VkFFTApp(d.shape, d.dtype, self.queue, ndim=ndim, norm=norm, dct=dct)

                                d = dctn(d, axes=list(range(dims))[-ndim:], type=dct)
                                d_gpu = app.fft(d_gpu)
                                self.assertTrue(np.allclose(d, d_gpu.get(), rtol=rtol, atol=abs(d).max() * rtol))

                                d = idctn(d, axes=list(range(dims))[-ndim:], type=dct)
                                d_gpu = app.ifft(d_gpu)

                                self.assertTrue(np.allclose(d, d_gpu.get(), rtol=rtol, atol=abs(d).max() * rtol))
                                self.assertTrue(np.allclose(d0, d_gpu.get(), rtol=rtol, atol=abs(d).max() * rtol))

    @unittest.skipIf(cla is None, "pyopencl is not available")
    @unittest.skipIf(not has_scipy_dct, "scipy.fft.dctn is not available")
    def test_dct_outofplace(self):
        """
        Test out-of-place DCT transforms
        """
        for n in self.vn:
            # test both radix-2 and Bluestein algorithms.
            max_dim = 4
            if n > 64:
                # Only test 1D and 2D for large sizes
                max_dim = 2
            for dims in range(1, max_dim + 1):
                for ndim in range(1, min(dims, 3) + 1):
                    sh = [n] * dims
                    for i in range(ndim, dims):
                        # Smaller array on non-transformed axes
                        sh[-i - 1] = 2
                    d0 = np.random.uniform(0, 1, sh)
                    # A pure random array may not be a very good test (too random),
                    # so add a Gaussian
                    xx = [np.fft.fftshift(np.fft.fftfreq(nn)) for nn in sh]
                    v = np.zeros_like(d0)
                    for x in np.meshgrid(*xx, indexing='ij'):
                        v += x ** 2
                    d0 += 10 * np.exp(-v * 2)
                    for dtype in self.dtype_float_v:
                        for dct in range(1, 4 + 1):
                            if dct == 1 and n > 750 * 4 // np.dtype(dtype).itemsize:
                                # Not sure where this limitation comes from
                                # limit is n > 757 for float64, 1532 for float32 on a GTX 1080
                                continue
                            elif n > 2000 * 4 // np.dtype(dtype).itemsize:
                                # limit is n > 2002 for float64, 4004 for float32 on a GTX 1080
                                continue
                            # for norm in [0, 1, "ortho"] # TODO : test all norms
                            norm = 1  # Same default norm as scipy
                            with self.subTest(n=n, dims=dims, ndim=ndim, dtype=dtype, dct=dct, norm=norm):
                                if dtype == np.float32:
                                    rtol = 1e-4
                                else:
                                    rtol = 1e-8
                                if max(primes(n)) > 13 or dct != 2:
                                    # Lower accuracy for Bluestein transforms
                                    # Why is it necessary for DCT types != 2 ?
                                    rtol *= 8

                                d = d0.astype(dtype).copy()

                                d_gpu = cla.to_device(self.queue, d)
                                d1_gpu = cla.empty_like(d_gpu)
                                app = VkFFTApp(d.shape, d.dtype, self.queue, ndim=ndim, norm=norm,
                                               dct=dct, inplace=False)

                                d = dctn(d, axes=list(range(dims))[-ndim:], type=dct)
                                d1_gpu = app.fft(d_gpu, d1_gpu)
                                self.assertTrue(np.allclose(d0, d_gpu.get(), rtol=rtol, atol=abs(d).max() * rtol))
                                self.assertTrue(np.allclose(d, d1_gpu.get(), rtol=rtol, atol=abs(d).max() * rtol))

                                d = idctn(d, axes=list(range(dims))[-ndim:], type=dct)
                                d_gpu = app.ifft(d1_gpu, d_gpu)

                                self.assertTrue(np.allclose(d, d_gpu.get(), rtol=rtol, atol=abs(d).max() * rtol))
                                self.assertTrue(np.allclose(d0, d_gpu.get(), rtol=rtol, atol=abs(d).max() * rtol))


def suite():
    test_suite = unittest.TestSuite()
    load_tests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(load_tests(TestVkFFTOpenCL))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
