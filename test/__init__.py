import unittest

from .test_cuda import suite as test_cuda_suite
from .test_opencl import suite as test_opencl_suite
from .test_fft import suite as test_fft_suite


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(test_cuda_suite())
    test_suite.addTest(test_opencl_suite())
    test_suite.addTest(test_fft_suite())
    return test_suite
