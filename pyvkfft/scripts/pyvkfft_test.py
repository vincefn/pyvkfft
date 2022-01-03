#!/Users/vincent/dev/py38-env/bin/python
# -*- coding: utf-8 -*-

# PyVkFFT
#   (c) 2022- : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr
#
#
# pyvkfft script to run short or long unit tests

import argparse
import sys
import unittest
import numpy as np
from pyvkfft.test import TestFFT, TestFFTSystematic


def main():
    parser = argparse.ArgumentParser(prog='pyvkfft-test',
                                     description='Run pyvkfft unittest, short or long (systematic)')
    parser.add_argument('-b', '--backend', action='store', nargs='*',
                        help="single or multiple GPU backends",
                        choices=['pycuda', 'cupy', 'pyopencl', 'all'],
                        default='all')

    args = parser.parse_args()

    print(args)
    # run_tests()


def suite_default():
    suite = unittest.TestSuite()
    load_tests = unittest.defaultTestLoader.loadTestsFromTestCase
    suite.addTest(load_tests(TestFFT))
    return suite


def suite_systematic():
    suite = unittest.TestSuite()
    load_tests = unittest.defaultTestLoader.loadTestsFromTestCase
    suite.addTest(load_tests(TestFFTSystematic))
    return suite


if __name__ == '__main__':
    # main()
    parser = argparse.ArgumentParser(prog='pyvkfft-test',
                                     description='Run pyvkfft unit tests, regular or systematic')
    parser.add_argument('--mailto', action='store',
                        help="Email address the results will be sent to")
    parser.add_argument('--mailto-fail', action='store',
                        help="Email address the results will be sent to, only if the test fails")
    parser.add_argument('--systematic', action='store_true',
                        help="Perform a systematic accuracy test over a range of array sizes.\n"
                             "Without this argument a faster test (a few minutes) will be "
                             "performed with selected array sizes for all possible transforms.")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Verbose output")
    sysgrp = parser.add_argument_group("systematic", "Options for --systematic:")
    sysgrp.add_argument('--axes', action='store', nargs='*', type=int,
                        help="transform axes: x (fastest) is 1,"
                             "y is 2, z is 3, e.g. '--axes 1', '--axes 2 3'."
                             "The default is to perform the transform along the ndim fastest "
                             "axes. Using this overrides --ndim")
    sysgrp.add_argument('--backend', action='store', nargs='*',
                        help="Choose single or multiple GPU backends,"
                             "'all' (the default) will automatically select the available ones.",
                        choices=['pycuda', 'cupy', 'pyopencl'])
    sysgrp.add_argument('--bluestein', action='store_true',
                        help="Only perform transform with non-radix dimensions, i.e. the "
                             "largest number in the prime decomposition of each array dimension "
                             "must be larger than 13")
    sysgrp.add_argument('--db', nargs='*', action='store',
                        help="Save the results to an sql database. If no filename is"
                             "given, pyvkfft-test.sql will be used. If the file already"
                             "exists, the results are added to the file. Fields stored"
                             "include HOSTNAME, EPOCH, BACKEND, LANGUAGE, TRANSFORM (c2c, r2c or "
                             "dct1/2/3/4, AXES, ARRAY_SHAPE, NDIMS, NDIM, PRECISION, INPLACE,"
                             "NORM, LUT, N, ACCURACY_FFT, ACCURACY_IFFT, TOLERANCE,"
                             "DT_APP, DT_FFT, DT_IFFT, SRC_UNCHANGED_FFT, SRC_UNCHANGED_IFFT, "
                             "GPU_NAME, SUCCESS, ERROR, VKFFT_ERROR_CODE")
    sysgrp.add_argument('--dct', nargs='*', action='store', type=int,
                        help="Test direct cosine transforms (default is c2c):"
                             " '--dct' (defaults to dct 2), '--dct 1'",
                        choices=[None, 1, 2, 3, 4])
    sysgrp.add_argument('--double', action='store_true',
                        help="Use double precision (float64/complex128) instead of single")
    sysgrp.add_argument('--dry-run', action='store_true',
                        help="Perform a dry-run, returning the number of tests to perform")
    sysgrp.add_argument('--inplace', action='store_true',
                        help="Use inplace transforms (NB: for R2C with ndim>=2, the x-axis "
                             "must be even-sized)")
    sysgrp.add_argument('--lut', action='store_true',
                        help="Force the use of a LUT for the transform, to improve accuracy. "
                             "By default VkFFT will activate the LUT on some GPU with less "
                             "accurate accelerated trigonometric functions. "
                             "This is automatically true for double precision")
    sysgrp.add_argument('--ndim', action='store', nargs=1,
                        help="Number of dimensions for the transform",
                        default=[1], type=int, choices=[1, 2, 3])
    sysgrp.add_argument('--ndims', action='store', nargs=1,
                        help="Number of dimensions for the array (must be >=ndim). "
                             "By default, the array will have the same dimensionality "
                             "as the transform (ndim)",
                        type=int, choices=[1, 2, 3, 4])
    sysgrp.add_argument('--norm', action='store', nargs=1, type=int,
                        help="Normalisation to test (must be 1 for dct)",
                        default=[1], choices=[0, 1])
    sysgrp.add_argument('--nproc', action='store', nargs=1,
                        help="Number of parallel process to use to speed up tests. "
                             "Make sure the sum of parallel process will not use too much "
                             "GPU memory",
                        default=['1'], type=int)
    sysgrp.add_argument('--r2c', action='store_true', help="Test real-to-complex transform "
                                                           "(default is c2c)")
    sysgrp.add_argument('--radix', action='store', nargs='*', type=int,
                        help="Perform only radix transforms. If no value is given, all available "
                             "radix transforms are allowed. Alternatively a list can be given: "
                             "'--radix 2' (only 2**n array sizes), '--radix 2 3 5' "
                             "(only 2**N1 * 3**N2 * 5**N3)",
                        choices=[None, 2, 3, 5, 7, 11, 13])
    sysgrp.add_argument('--range', action='store', nargs=2, type=int,
                        help="Range of array sizes [min, max] along each transform dimension, "
                             "'--range 2 128'",
                        default=[2, 128])

    parser.print_help()
    args = parser.parse_args()
    print(args)

    if args.systematic:
        TestFFTSystematic.axes = args.axes
        TestFFTSystematic.bluestein = args.bluestein
        TestFFTSystematic.dct = False if args.dct is None else args.dct[0] if len(args.dct) else 2
        TestFFTSystematic.db = args.db
        TestFFTSystematic.dry_run = args.dry_run
        TestFFTSystematic.dtype = np.float64 if args.double else np.float32
        TestFFTSystematic.inplace = args.inplace
        TestFFTSystematic.lut = args.lut
        TestFFTSystematic.ndim = args.ndim[0]
        TestFFTSystematic.ndims = args.ndims
        TestFFTSystematic.norm = args.norm[0]
        TestFFTSystematic.nproc = args.nproc[0]
        TestFFTSystematic.r2c = args.r2c
        TestFFTSystematic.radix = args.radix
        TestFFTSystematic.range = args.range
        TestFFTSystematic.vbackend = args.backend
        TestFFTSystematic.verbose = args.verbose
        TestFFTSystematic.vn = args.range
        if args.verbose:
            unittest.main(defaultTest='suite_systematic', verbosity=2, argv=sys.argv[:1])
        else:
            unittest.main(defaultTest='suite_systematic', verbosity=1, argv=sys.argv[:1])
    else:
        if args.verbose:
            unittest.main(defaultTest='suite_default', verbosity=2, argv=sys.argv[:1])
        else:
            unittest.main(defaultTest='suite_default', verbosity=1, argv=sys.argv[:1])
