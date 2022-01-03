import unittest

from .test_fft import suite as test_fft_suite, TestFFT, TestFFTSystematic


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(test_fft_suite())
    return test_suite
