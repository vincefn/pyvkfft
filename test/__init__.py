import unittest

from .test_cuda import suite as test_cuda_suite


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(test_cuda_suite())
    return test_suite
