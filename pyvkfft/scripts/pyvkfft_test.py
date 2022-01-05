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
import os.path
import sys
import unittest
import numpy as np
from pyvkfft.test import TestFFT, TestFFTSystematic


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


def main():
    parser = argparse.ArgumentParser(prog='pyvkfft-test',
                                     description='Run pyvkfft unit tests, regular or systematic')
    parser.add_argument('--colour', action='store_true',
                        help="Use colour depending on how good the measured accuracy is")
    parser.add_argument('--mailto', action='store',
                        help="Email address the results will be sent to")
    parser.add_argument('--mailto_fail', action='store',
                        help="Email address the results will be sent to, only if the test fails")
    parser.add_argument('--mailto_smtp', action='store', default="localhost",
                        help="SMTP server address to mail the results")
    parser.add_argument('--silent', action='store_true',
                        help="Use this to minimise the written output "
                             "(note that tests can take a long time be patient")
    parser.add_argument('--systematic', action='store_true',
                        help="Perform a systematic accuracy test over a range of array sizes.\n"
                             "Without this argument a faster test (a few minutes) will be "
                             "performed with selected array sizes for all possible transforms.")
    sysgrp = parser.add_argument_group("systematic", "Options for --systematic:")
    sysgrp.add_argument('--axes', action='store', nargs='*', type=int,
                        help="transform axes: x (fastest) is 1,"
                             "y is 2, z is 3, e.g. '--axes 1', '--axes 2 3'."
                             "The default is to perform the transform along the ndim fastest "
                             "axes. Using this overrides --ndim")
    sysgrp.add_argument('--backend', action='store', nargs='+',
                        help="Choose single or multiple GPU backends,"
                             "by default all available backends are selected.",
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
                             "NORM, LUT, N, N2_FFT, N2_IFFT, NI_FFT, NI_IFFT, TOLERANCE,"
                             "DT_APP, DT_FFT, DT_IFFT, SRC_UNCHANGED_FFT, SRC_UNCHANGED_IFFT, "
                             "GPU_NAME, SUCCESS, ERROR, VKFFT_ERROR_CODE")
    sysgrp.add_argument('--dct', nargs='*', action='store', type=int,
                        help="Test direct cosine transforms (default is c2c):"
                             " '--dct' (defaults to dct 2), '--dct 1'",
                        choices=[1, 2, 3, 4])
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
                        default=[1], type=int)
    sysgrp.add_argument('--r2c', action='store_true', help="Test real-to-complex transform "
                                                           "(default is c2c)")
    sysgrp.add_argument('--radix', action='store', nargs='*', type=int,
                        help="Perform only radix transforms. If no value is given, all available "
                             "radix transforms are allowed. Alternatively a list can be given: "
                             "'--radix 2' (only 2**n array sizes), '--radix 2 3 5' "
                             "(only 2**N1 * 3**N2 * 5**N3)",
                        choices=[2, 3, 5, 7, 11, 13])
    sysgrp.add_argument('--range', action='store', nargs=2, type=int,
                        help="Range of array sizes [min, max] along each transform dimension, "
                             "'--range 2 128'",
                        default=[2, 128])

    # parser.print_help()
    args = parser.parse_args()

    # We modify class attributes to pass arguments - not a great approach but works..
    if args.systematic:
        t = TestFFTSystematic
        t.axes = args.axes
        t.bluestein = args.bluestein
        t.colour = args.colour
        t.dct = False if args.dct is None else args.dct[0] if len(args.dct) else 2
        t.db = args.db[0] if args.db is not None else None
        t.dry_run = args.dry_run
        t.dtype = np.float64 if args.double else np.float32
        t.inplace = args.inplace
        t.lut = args.lut
        t.ndim = args.ndim[0]
        t.ndims = args.ndims
        t.norm = args.norm[0]
        t.nproc = args.nproc[0]
        t.r2c = args.r2c
        t.radix = args.radix
        t.range = args.range
        t.vbackend = args.backend
        t.verbose = not args.silent
        t.vn = args.range
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(t)
        if t.verbose:
            res = unittest.TextTestRunner(verbosity=2).run(suite)
        else:
            res = unittest.TextTestRunner(verbosity=1).run(suite)
    else:
        t = TestFFT
        t.verbose = not args.silent
        t.colour = args.colour
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(t)
        if t.verbose:
            res = unittest.TextTestRunner(verbosity=2).run(suite)
        else:
            res = unittest.TextTestRunner(verbosity=1).run(suite)

    sub = os.path.split(sys.argv[0])[-1]
    for i in range(len(sys.argv)):
        arg = sys.argv[i]
        if 'mail' not in arg and 'mail' not in sys.argv[i - 1]:
            sub += " " + arg
    info = "Running:\n%s\n" % sub

    nb_err_fail = len(res.errors) + len(res.failures)
    if len(res.errors):
        info += "\nERRORS:\n\n"
        for t, s in res.errors:
            tid = t.id()
            tid1 = tid.split('.')[-1]
            tid0 = tid.split('.' + tid1)[0]
            info += '%s (%s):\n' % (tid1, tid0) + s
    if len(res.failures):
        info += "\nFAILURES:\n\n"
        for t, s in res.failures:
            tid = t.id()
            tid1 = tid.split('.')[-1]
            tid0 = tid.split('.' + tid1)[0]
            info += '%s (%s):\n\n' % (tid1, tid0) + s + '\n\n'

    if args.mailto_fail is not None and (nb_err_fail > 0) or args.mailto is not None:
        import smtplib
        try:
            from email.message import EmailMessage

            msg = EmailMessage()
            msg['From'] = 'favre@esrf.fr'
            msg['to'] = args.mailto if args.mailto is not None else args.mailto_fail
            msg['Subject'] = '[fail=%d error=%d] %s' % \
                             (len(res.failures), len(res.errors), sub)
            print("Mailing results:\nFrom: %s\nTo: %sSubject: %s" % (msg['to'], msg['to'], msg['Subject']))
            msg.set_content(info)

            s = smtplib.SMTP(args.mailto_smtp)
            s.send_message(msg)
            s.quit()
            print("Sent message with subject: %s" % msg['Subject'])
        except (ConnectionRefusedError, smtplib.SMTPConnectError):
            print("Could not connect to SMTP server (%s) to send email." % args.mailto_smtp)

    sys.exit(int(nb_err_fail > 0))


if __name__ == '__main__':
    main()
