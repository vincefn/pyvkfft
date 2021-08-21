import unittest
import numpy as np

try:
    from scipy.misc import ascent
except ImportError:
    def ascent():
        return np.random.randint(0, 255, (512, 512))

from pyvkfft.fft import fftn, ifftn, rfftn, irfftn

try:
    import pycuda.autoinit
    import pycuda.gpuarray as cua

    has_pycuda = True
except ImportError:
    has_pycuda = False

try:
    import cupy as cp

    has_cupy = True
except ImportError:
    has_cupy = False

try:
    import pyopencl as cl
    import pyopencl.array as cla
    import os

    # Create some context on the first available GPU
    if 'PYOPENCL_CTX' in os.environ:
        ctx = cl.create_some_context()
    else:
        ctx = None
        # Find the first OpenCL GPU available and use it, unless
        for p in cl.get_platforms():
            for d in p.get_devices():
                if d.type & cl.device_type.GPU == 0:
                    continue
                print("Selected device: ", d.name)
                ctx = cl.Context(devices=(d,))
                break
            if ctx is not None:
                break
    cq = cl.CommandQueue(ctx)

    has_pyopencl = True
except ImportError:
    has_pyopencl = False


class TestSimple(unittest.TestCase):

    @unittest.skipIf(not has_pycuda, "pycuda is not available")
    def test_pycuda(self):
        # C2C, new destination array
        d = cua.to_gpu(ascent().astype(np.complex64))
        d = fftn(d)
        d = ifftn(d)
        # in-place
        d = fftn(d, d)
        d = ifftn(d, d)
        # out-of-place
        d2 = cua.empty_like(d)
        d2 = fftn(d, d2)
        d = ifftn(d2, d)

        # R2C
        d = cua.to_gpu(ascent().astype(np.float32))
        d = rfftn(d)
        d = irfftn(d)

    @unittest.skipIf(not has_cupy, "cupy is not available")
    def test_cupy(self):
        # C2C, new destination array
        d = cp.array(ascent().astype(np.complex64))
        d = fftn(d)
        d = ifftn(d)
        # in-place
        d = fftn(d, d)
        d = ifftn(d, d)
        # out-of-place
        d2 = cp.empty_like(d)
        d2 = fftn(d, d2)
        d = ifftn(d2, d)

        # R2C
        d = cp.array(ascent().astype(np.float32))
        d = rfftn(d)
        d = irfftn(d)

    @unittest.skipIf(not has_pyopencl, "opencl is not available")
    def test_pyopencl(self):
        # C2C, new destination array
        d = cla.to_device(cq, ascent().astype(np.complex64))
        d = fftn(d)
        d = ifftn(d)
        # in-place
        d = fftn(d, d)
        d = ifftn(d, d)
        # out-of-place
        d2 = cla.empty_like(d)
        d2 = fftn(d, d2)
        d = ifftn(d2, d)

        # R2C
        d = cla.to_device(cq, ascent().astype(np.float32))
        d = rfftn(d)
        d = irfftn(d)


def suite():
    test_suite = unittest.TestSuite()
    load_tests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(load_tests(TestSimple))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
