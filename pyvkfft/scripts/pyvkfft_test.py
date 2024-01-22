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
import time
import timeit
import socket
import psutil
import numpy as np
from pyvkfft.test import TestFFT, TestFFTSystematic
from pyvkfft.version import __version__, vkfft_version, vkfft_git_version


def cuda_info_str():
    try:
        # The functions used should not create a context on the GPU
        from ..cuda import cuda_compile_version, cuda_runtime_version, cuda_driver_version
        return "CUDA: driver %s, runtime %s, compiled %s" % \
            (cuda_driver_version(), cuda_runtime_version(), cuda_compile_version())
    except:
        return ""


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


def make_html_pre_post(overwrite=False):
    if ('pyvkfft-test1000.html' not in os.listdir()) or overwrite:
        # if this was called by a parent pyvkfft-test-suite, print that command
        com = ""
        for c in psutil.Process(os.getppid()).cmdline():
            if "/" in c and "pyvkfft-test-suite" in c:
                c = os.path.split(c)[-1]
            com += "%s " % c
        if 'pyvkfft-test-suite' in com:
            com = 'Command:<tt>%s</tt>\n' % com
        else:
            com = ''
        # Need the html header, styles and the results' table beginning
        vkfft_ver = f'{vkfft_version()}' if 'unknown' in vkfft_git_version() \
            else f'{vkfft_version()}[{vkfft_git_version()}]'
        tmp = '<!DOCTYPE html>\n <html>\n <head> <style>\n' \
              'th, td { border: 1px solid grey;}\n' \
              '.center {margin-left: auto;  margin-right: auto; text-align:center;' \
              ' width:100%%; max-width:1920px; margin-left: auto;  margin-right: auto}\n' \
              '.cell_transform {background-color: #ccf;}\n' \
              '.active, .cell_transform:hover {background-color: #aaf;}\n' \
              '.toggle_graph {' \
              '  background-color: transparent;' \
              '  border: none;' \
              '  cursor: pointer;' \
              '  padding:0;' \
              '  outline: none;' \
              '  height: 100%%' \
              '  width: 100%%' \
              '}\n' \
              '.cell_error {background-color: #fcc;}\n' \
              '.active, .cell_error:hover {background-color: #faa;}\n' \
              '.toggle_error {' \
              '  background-color: transparent;' \
              '  border: none;' \
              '  cursor: pointer;' \
              '  padding:0;' \
              '  outline: none;' \
              '  height: 100%%' \
              '  width: 100%%' \
              '}\n' \
              '.toggle_fail {' \
              '  background-color: transparent;' \
              '  border: none;' \
              '  cursor: pointer;' \
              '  padding:0;' \
              '  outline: none;' \
              '  height: 100%%' \
              '  width: 100%%' \
              '}\n' \
              '.label_ok {' \
              'background-color: #00ff00;' \
              'font-weight: bold;' \
              'color: #000;' \
              '}\n' \
              '</style>\n' \
              '</head>\n' \
              '<body>\n' \
              '<div class="center">' \
              '<h2>pyVkFFT test results</h2>\n' \
              '<h3>pyvkfft: %s,  VkFFFT:%s %s host : %s</h3>\n' \
              '<div style="text-align:left;">' \
              '<p>%s<p>Methodology: the included graphs measure the accuracy of the forward ' \
              'and backward transforms: an array is generated with random uniform values ' \
              'between -0.5 and 0.5, and the results of its transform are compared ' \
              'with either pyfftw (in long double precision) if available, or scipy if ' \
              'available, or numpy fft. The L2 curve measures the average square norm ' \
              'difference, and the L<sub>&infin;</sub> the maximum difference.' \
              '<p>Note: for the R2C inverse transform, the result of the forward ' \
              'transform is used instead of re-using the random array (in order to have ' \
              'a proper half-Hermitian array), contrary to what is done for other ' \
              'transforms. This explains why the IFFT R2C maximum (L<sub>&infin;</sub>) ' \
              'errors are larger.' \
              '<p>Note 2: some "errors" for DCT may be due to unsupported sizes in VkFFT, ' \
              'which vary depending on the card and language used (amount of ' \
              'shared/local memory). So they just indicate a current limit for the ' \
              'transform sizes rather than a real error.' \
              '<p> [Click on the highlighted cells for details and accuracy graphs ' \
              'vs the transform size]<br>\n' \
              '</div><table class="center">\n' \
              '   <thead>\n' \
              '       <tr>\n' \
              '           <th>GPU</th>' \
              '           <th>backend</th>' \
              '           <th>transform</th>' \
              '           <th>ndim</th>' \
              '           <th>range</th>' \
              '           <th>radix</th>' \
              '           <th>dtype</th>' \
              '           <th>inplace</th>' \
              '           <th>LUT</th>' \
              '           <th>norm</th>' \
              '           <th>time-duration</th>' \
              '           <th>FAIL</th>' \
              '           <th>ERROR</th>' \
              '       </tr>\n' \
              '   </thead>\n' \
              '<tbody class="center">\n' % (__version__, vkfft_ver, cuda_info_str(),
                                            socket.gethostname(), com)
        open("pyvkfft-test1000.html", "w").write(tmp)
    if ('pyvkfft-test1999.html' not in os.listdir()) or overwrite:
        tmp = '</tbody>\n' \
              '</tbody\n' \
              '</table>\n' \
              '</div>\n' \
              '  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>' \
              '     <script id="rendered-js" >\n' \
              '        $(document).ready(function ()' \
              '        {\n' \
              '           $(".toggle_graph").click(function () \n' \
              '              {\n' \
              '                 var trgraph = $(this).parents().nextUntil(".row_results", ".graph"); \n' \
              '                 trgraph.toggle(); \n' \
              '                 if(trgraph.is( ":visible" ))\n' \
              '                 { trgraph.children("td:first").html($(this).data("command")+"<br>' \
              '                      <img src=\\"" + $(this).data("img") + "\\">");}\n' \
              '                 else { trgraph.children("td:first").html("img hidden");};\n' \
              '                 \n' \
              '               });\n' \
              '           $(".toggle_fail").click(function () ' \
              '              {$(this).parents().nextUntil(".row_results", ".failures").toggle();  });\n' \
              '           $(".toggle_error").click(function () ' \
              '              {$(this).parents().nextUntil(".row_results", ".errors").toggle();  });\n' \
              '        });\n' \
              '</script>' \
              '' \
              '</body>\n' \
              '</html>'
        open("pyvkfft-test1999.html", "w").write(tmp)


def name_next_file(pattern="pyvkfft-test%04d.html"):
    """
    Find the first unused name for a file, starting at i=1
    :param pattern: the pattern for the file name.
    :return: the filename
    """
    lsdir = os.listdir()
    for i in range(1001, 1999):
        if pattern % i not in lsdir:
            return pattern % i
    raise RuntimeError("name_next_file: '%s' files all used from 1001 to 1998. Maybe cleanup ?" % pattern)


def make_parser():
    epilog = "Examples:\n" \
             "   pyvkfft-test\n" \
             "      the regular test which tries the fft interface, using parallel\n" \
             "      streams (for pycuda), and C2C/R2C/DCT/DST transforms for sizes N=15,17,30,34\n" \
             "      with 1D to 4 or 5D transforms, also N=808,2988,4200,13000,13001," \
             "      13002,130172 for 1D and 2D transforms.\n" \
             "      All tests are done with single and double precision, in and\n" \
             "      out-of-place, norm=0 and 1, and all available backends (pyopencl,\n" \
             "      pycuda and cupy). For C2C arrays up to dimension 5 are tested,\n" \
             "      with all possible combination of transform axes.\n" \
             "      That's for a total of a tens of thousands transforms, which are tested\n" \
             "      against the result of numpy, scipy or pyfftw (when available) for\n" \
             "      accuracy.\n" \
             "      The text output gives the N2 and Ninf (aka max) relative norm of\n" \
             "      the transform, with the ratio in () to the expected tolerance for\n" \
             "      both direct and inverse transforms.\n" \
             "\n" \
             "  pyvkfft-test --nproc 8 --gpu v100 --mailto_fail toto@pyvkfft.org\n" \
             "      same test, but using 8 parallel process to speed up, and use a GPU\n" \
             "      with 'v100' in its name. Also, send the results in case of a failure\n" \
             "      to the given email address\n" \
             "\n" \
             "  pyvkfft-test --systematic --backend pycuda --nproc 8 --radix --range 2 10000\n" \
             "      Perform a systematic test of C2C transforms in (by default) 1D and\n" \
             "      single precision, for N=2 to 10000, only for radix transforms\n" \
             "\n" \
             "  pyvkfft-test --systematic --backend pycuda --nproc 8 --radix 2 7 11 --range 2 10000 --double\n" \
             "      Same test, but only for radix sizes with factors 2, 7 and 11, and double accuracy\n" \
             "\n" \
             "  pyvkfft-test --systematic --backend cupy --nproc 8 --bluestein --range 2 10000 --ndim 2 " \
             "--lut --inplace\n" \
             "      test with cupy backend, only non-radix 2D inplace R2C transforms\n," \
             "      using a lookup table( lut) for higher single precision accuracy.\n\n\n" \
             "Columns in the text output:\n" \
             " * backend\n" \
             " * type of transform\n" \
             " * array shape\n" \
             " * axes for the transform. If None, axes are set by the number of transform dimensions\n" \
             " * number of dimensions for the transform. Can be None if axes are given.\n" \
             " * type of algorithm for each axis: r=radix, R=Rader, B=Bluestein, -=skipped axis\n" \
             " * number of uploads for each axis: 0 if not transformed, 1 if the axis length fits\n" \
             "   in the cache and the transform can be done in 1 read+write, 2 or 3 if multi-upload is used\n" \
             " * data type and precision\n" \
             " * use of a Look-Up-Table (LUT) -for single precision only.\n" \
             " * inplace or out-of-place transform\n" \
             " * normalisation for the transform: 0 or 1\n" \
             " * order of the array: C (fast axis is last) or F (fast axis is first)\n" \
             " * N2 and N_inf error norm for the forward transform, with the comparison \n" \
             "   to the maximum allowed error (and in parenthesis the ratio to this maximum), \n" \
             "   and finally 0 or 1 depending on whether the source array was modified (0) or not (1)\n" \
             " * Same values for the inverse transform\n" \
             " * temporary buffer size allocated by VkFFT if necessary, for large transforms\n" \
             " * status: OK, FAIL (if accuracy is above limit or source array unexpectedly changed)\n" \
             "   or ERROR (an error was raised during execution, e.g. compilation, memory,...)\n"
    parser = argparse.ArgumentParser(prog='pyvkfft-test', epilog=epilog,
                                     description='Run pyvkfft unit tests, regular or systematic',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--colour', action='store_true',
                        help="Use colour depending on how good the measured accuracy is")
    parser.add_argument('--html', action='store', nargs='*',
                        help="Summarises the results in html row(s). This is saved to "
                             "'pyvkfft-test%%04d.html', starting at i=1001 and incrementing. "
                             "Files with i=1000 and i=1999 are the beginning and the end of the"
                             "html file, which can be concatenated to form a valid html page."
                             "If --graph is also used, this includes a graph of the accuracy "
                             "which can be displayed by clicking on the type of transform.")
    parser.add_argument('--gpu', action='store',
                        help="Name (or sub-string) of the GPU to use")
    parser.add_argument('--opencl_platform', action='store',
                        help="Name (or sub-string) of the opencl platform to use (case-insensitive). "
                             "Note that by default the PoCL platform is skipped, "
                             "unless it is specifically requested or it is the only one available "
                             "(PoCL has some issues with VkFFT for some transforms)")
    parser.add_argument('--mailto', action='store',
                        help="Email address the results will be sent to")
    parser.add_argument('--mailto_fail', action='store',
                        help="Email address the results will be sent to, only if the test fails")
    parser.add_argument('--mailto_smtp', action='store', default="localhost",
                        help="SMTP server address to mail the results")
    parser.add_argument('--nproc', action='store', nargs=1,
                        help="Number of parallel process to use to speed up tests. "
                             "Make sure the sum of parallel process will not use too much "
                             "GPU memory",
                        default=[1], type=int)
    parser.add_argument('--silent', action='store_true',
                        help="Use this to minimise the written output "
                             "(note that tests can take a long time be patient")
    parser.add_argument('--c2c', action='store_true',
                        help="When used without --systematic, perform only "
                             "c2c quick tests and skip the long r2c/dct/dst "
                             "unless they were also requested.")
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
    sysgrp.add_argument('--bluestein', '--nonradix', action='store_true',
                        help="Only perform transform with non-radix dimensions, i.e. the "
                             "largest number in the prime decomposition of each array dimension "
                             "must be larger than 13")
    sysgrp.add_argument('--db', nargs='*', action='store',
                        help="Save the results to an sql database. If no filename is"
                             "given, pyvkfft-test.sql will be used. If the file already"
                             "exists, the results are added to the file. Fields stored"
                             "include HOSTNAME, EPOCH, BACKEND, LANGUAGE, TRANSFORM (c2c, r2c or "
                             "dct/dst1/2/3/4, AXES, ARRAY_SHAPE, NDIMS, NDIM, PRECISION, INPLACE,"
                             "NORM, LUT, N, N2_FFT, N2_IFFT, NI_FFT, NI_IFFT, TOLERANCE,"
                             "DT_APP, DT_FFT, DT_IFFT, SRC_UNCHANGED_FFT, SRC_UNCHANGED_IFFT, "
                             "GPU_NAME, SUCCESS, ERROR, VKFFT_ERROR_CODE")
    sysgrp.add_argument('--dct', nargs='?', action='store', type=int,
                        help="Test direct cosine transforms (default is c2c):"
                             " '--dct' (defaults to dct 2), '--dct 1'",
                        const=2, default=False, choices=[1, 2, 3, 4])
    sysgrp.add_argument('--dst', nargs='?', action='store', type=int,
                        help="Test direct sine transforms (default is c2c):"
                             " '--dst' (defaults to dst 2), '--dst 1'",
                        const=2, default=False, choices=[1, 2, 3, 4])
    sysgrp.add_argument('--double', action='store_true',
                        help="Use double precision (float64/complex128) instead of single")
    sysgrp.add_argument('--dry-run', action='store_true',
                        help="Perform a dry-run, printing the number of array shapes to test")
    parser.add_argument('--fast-random', action='store', default=None,
                        help="Use this option to run a random percentage of the full test suite, "
                             "for faster results. A number between 5 and 100 is required.", type=int)
    sysgrp.add_argument('--inplace', action='store_true',
                        help="Use inplace transforms")
    sysgrp.add_argument('--graph', action='store', nargs="?", const="",
                        help="Save the graph of the accuracy as a function of the size"
                             "to the given filename (if no name is given, it will be "
                             "automatically generated)."
                             "Requires matplotlib, and scipy for linear regression.")
    sysgrp.add_argument('--lut', action='store_true',
                        help="Force the use of a LUT for the transform, to improve accuracy. "
                             "By default VkFFT will activate the LUT on some GPU with less "
                             "accurate accelerated trigonometric functions. "
                             "This is automatically true for double precision")
    sysgrp.add_argument('--max-nb-tests', action='store', nargs=1,
                        help="Maximum number of tests. If the number of generated test "
                             "cases is larger, the program will abort.",
                        default=[1000], type=int)
    sysgrp.add_argument('--ndim', action='store', nargs=1,
                        help="Number of dimensions for the transform. Using 12 or 123 "
                             "will result in testing bother 1 and 2 or 1,2 and 3. It is"
                             "recommended to use --range_mb and ",
                        default=[1], type=int, choices=[1, 2, 3, 12, 123])
    # sysgrp.add_argument('--ndims', action='store', nargs=1,
    #                     help="Number of dimensions for the array (must be >=ndim). "
    #                          "By default, the array will have the same dimensionality "
    #                          "as the transform (ndim)",
    #                     type=int, choices=[1, 2, 3, 4])
    sysgrp.add_argument('--norm', action='store', nargs=1, type=int,
                        help="Normalisation to test (must be 1 for dct or dst)",
                        default=[1], choices=[0, 1])
    sysgrp.add_argument('--ref-long-double', action='store_true',
                        help="Use long double precision for the reference calculation, "
                             "(requires scipy). This gives more objective accuracy plots but "
                             "can be slower (or much slower on some architectures).")
    sysgrp.add_argument('--r2c', action='store_true', help="Test real-to-complex transform "
                                                           "(default is c2c)")
    sysgrp.add_argument('--fstride', action='store_true',
                        help="Test F-ordered arrays (default is C-ordered). Not supported for DCT/DST")
    sysgrp.add_argument('--radix', action='store', nargs='*', type=int,
                        help="Perform only radix transforms. If no value is given, all available "
                             "radix transforms are allowed. Alternatively a list can be given: "
                             "'--radix 2' (only 2**n array sizes), '--radix 2 3 5' "
                             "(only 2**N1 * 3**N2 * 5**N3)",
                        choices=[2, 3, 5, 7, 11, 13])
    sysgrp.add_argument('--radix-max-pow', action='store', nargs=1, type=int,
                        help="For radix runs, specify the maximum exponent of each base "
                             "integer, i.e. for '--radix 2 3 --radix-max-pow 2' will limit "
                             "lengths to 2**N1 * 3**N2 with N1,N2<=2")
    sysgrp.add_argument('--range', action='store', nargs=2, type=int,
                        help="Range of array lengths [min, max] along each transform dimension, "
                             "'--range 2 128'",
                        default=[2, 128])
    sysgrp.add_argument('--range-mb', action='store', nargs=2, type=int,
                        help="Allowed range of array sizes [min, max] in Mbytes, e.g. "
                             "'--range-mb 2 128'. This can be used to limit the arrays size "
                             "while allowing large lengths along individual dimensions. "
                             "It can also be used to separate runs with a given size range "
                             "and different nproc values. This takes into account the "
                             "type (single or double), and also whether the transform "
                             "is made inplace, so this represents the total GPU memory"
                             "used.",
                        default=[0, 128])
    sysgrp.add_argument('--range-nd-narrow', action='store', nargs=2, default=['0', '0'],
                        help="Two values (drel dabs), e.g. '--range_nd_narrow 0.10 11' "
                             "with 0<=drel<=1 and dabs (integer>=0) must be given "
                             "to allow 2D and 3D tests to be done on arrays with different "
                             "lengths along every dimension, but while limiting the "
                             "difference between lengths. For example in 2D for an "
                             "(N1,N2) array shape, generated lengths will verify "
                             "abs(n2-n1)<max(dabs+drel*N1). The default value of (0,0) "
                             "only allows the same lengths. This allows to test more "
                             "diverse configurations while limiting the number of tests.")
    sysgrp.add_argument('--serial', action='store_true',
                        help="Serialise the tests instead of spawning them in separate "
                             "process, allowing to diagnose more errors. Incompatible "
                             "with nproc>1.")
    sysgrp.add_argument('--timeout', action='store', nargs=1, type=int,
                        help="Change the timeout (in seconds) to raise a TimeOut error for "
                             "individual tests. After 4 have failed, give up.",
                        default=[120])
    return parser


def main():
    t0 = timeit.default_timer()
    localt0 = time.localtime()
    parser = make_parser()

    # parser.print_help()
    args = parser.parse_args()
    if args.serial and args.nproc[0] > 1:
        raise RuntimeError("Cannot use --serial with --nproc")
    if args.graph is not None:
        if not len(args.graph):
            args.graph = name_next_file("pyvkfft-test%03d.svg")

    # We modify class attributes to pass arguments - not a great approach but works..
    if args.systematic:
        t = TestFFTSystematic
        t.axes = args.axes
        t.bluestein = args.bluestein
        t.colour = args.colour
        t.dct = args.dct
        t.dst = args.dst
        t.db = args.db[0] if args.db is not None else None
        t.dry_run = args.dry_run
        t.dtype = np.float64 if args.double else np.float32
        t.fast_random = args.fast_random
        if args.fast_random is not None:
            assert 5 <= args.fast_random <= 100, "--fast-random must be between 5 and 100"
        t.gpu = args.gpu
        t.opencl_platform = args.opencl_platform
        t.graph = args.graph
        t.inplace = args.inplace
        t.lut = args.lut
        t.max_nb_tests = args.max_nb_tests[0]
        t.ndim = args.ndim[0]
        # t.ndims = args.ndims
        t.norm = args.norm[0]
        t.nproc = args.nproc[0]
        t.opencl_platform = args.opencl_platform
        t.r2c = args.r2c
        t.fstride = args.fstride
        t.radix = args.radix
        t.ref_long_double = args.ref_long_double
        t.max_pow = None if args.radix_max_pow is None else args.radix_max_pow[0]
        t.range = args.range
        size_min_max = np.array(args.range_mb) * 1024 ** 2 // 8
        if args.r2c or args.dct or args.dst:
            size_min_max = size_min_max * 2
        if args.double:
            size_min_max = size_min_max / 2
        if args.inplace:
            size_min_max = size_min_max / 2
        t.range_size = size_min_max.tolist()
        t.range_nd_narrow = float(args.range_nd_narrow[0]), int(args.range_nd_narrow[1])
        t.serial = args.serial
        t.timeout = args.timeout[0]
        t.vbackend = args.backend
        t.verbose = not args.silent
        t.vn = args.range
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(t)
        if t.verbose:
            res = unittest.TextTestRunner(verbosity=2).run(suite)
        else:
            res = unittest.TextTestRunner(verbosity=1).run(suite)
        if t.dry_run:
            print(t.nb_shapes_gen)
            sys.exit()
    else:
        t = TestFFT
        t.verbose = not args.silent
        t.colour = args.colour
        t.gpu = args.gpu
        t.nproc = args.nproc[0]
        t.opencl_platform = args.opencl_platform
        t.vbackend = args.backend
        if args.backend is not None:
            # Remove tests depending on backend used
            test_cuda_stream = False
            for b in args.backend:
                if 'pycuda' in b.lower():
                    test_cuda_stream = True
            if not test_cuda_stream:
                del t.test_pycuda_streams
            if 'pyopencl' not in args.backend:
                del t.test_pyopencl_queues
        if args.r2c or args.dct or args.dst or args.c2c:
            # A selection of subtests was made - remove the other long ones
            if not args.r2c:
                del t.test_r2c
            if not args.dct:
                del t.test_dct
            if not args.dst:
                del t.test_dst
            if not args.c2c:
                del t.test_c2c
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(t)
        if t.verbose:
            res = unittest.TextTestRunner(verbosity=2).run(suite)
        else:
            res = unittest.TextTestRunner(verbosity=1).run(suite)

    sub = os.path.split(sys.argv[0])[-1]
    for i in range(1, len(sys.argv)):
        arg = sys.argv[i]
        if 'mail' not in arg and 'mail' not in sys.argv[i - 1] and 'html' not in arg and 'graph' not in arg:
            sub += " " + arg
    info = "Ran:\n   %s\n\n Result:%s\n\n" % (sub, "OK" if res.wasSuccessful() else "FAILURE")

    info += "Elapsed time for tests: %s\n\n" % time.strftime("%Hh %Mm %Ss", time.gmtime(timeit.default_timer() - t0))

    nb_err_fail = len(res.errors) + len(res.failures)

    if args.html is not None:
        make_html_pre_post(overwrite=False)
        html = ''
        # One row for the summary
        html += '<tr class="row_results">'
        html += '<td>%s</td><td>' % (args.gpu if args.gpu is not None else '-')
        if args.backend is not None:
            for a in args.backend:
                html += a + ' '
        else:
            html += 'all'
        html += '</td>'
        if args.systematic:
            has_graph = False
            if args.graph is not None:
                if os.path.exists(args.graph):
                    has_graph = True
            if has_graph:
                tmp = '<td class="cell_transform"><input class="toggle_graph" type="button"' \
                      'data-img="%s" data-command="%s"' % (args.graph, sub)
                tmp += 'value="%s" style="width:100%%; height:100%%"></td>'
            else:
                tmp = '<td>%s</td>'
            if args.r2c:
                html += tmp % 'R2C'
            elif t.dct or t.dst:
                html += tmp % ('DCT%d' % t.dct) if t.dct else tmp % ('DST%d' % t.dst)
            else:
                html += tmp % 'C2C'
            if t.ndim == 12:
                html += "<td>1,2</td>"
            elif t.ndim == 123:
                html += "<td>1,2,3</td>"
            else:
                html += "<td>%d</td>" % t.ndim
            html += "<td>%d-%d" % (args.range[0], args.range[1])
            if (t.range_nd_narrow[0] > 0 or t.range_nd_narrow[1] > 0) and t.ndim > 1:
                html += " <small>[|N<sub>i</sub>-N<sub>1</sub>|<={%d; %d%%N<sub>1</sub>}]</small>" % \
                        (t.range_nd_narrow[1], int(t.range_nd_narrow[0] * 100))
            elif t.ndim > 1:
                html += ' (' + 'N,' * (t.ndim - 1) + 'N)'
            html += "</td>"
            if args.bluestein:
                html += "<td>Bluestein</td>"
            elif args.radix is None:
                html += "<td>-</td>"
            else:
                html += "<td>"
                for i in (args.radix if len(args.radix) else [2, 3, 5, 7, 11, 13]):
                    html += "%d," % i
                html = html[:-1]
                if t.max_pow is not None:
                    html += '<small>[^N,N<=%d]</small>' % t.max_pow
                html += '</td>'
            html += "<td>%s</td>" % ('float64' if args.double else 'float32')
            html += "<td>%s</td>" % ('inplace' if args.inplace else 'out-of-place')
            html += "<td>%s</td>" % ('True' if args.lut else 'Auto')
            html += "<td>%d</td>" % args.norm[0]
            nbts = '[%5d tests]' % t.nb_test
        else:
            html += ''
            html += '<td colspan="8">Regular multi-dimensional C2C/R2C/R2R test</td>'
            nbts = ''

        html += '<td>%s +%s %s</td>' % (time.strftime("%Y-%m-%d %Hh%M:%S", localt0),
                                        time.strftime("%Hh %Mm %Ss", time.gmtime(timeit.default_timer() - t0)), nbts)

        if len(res.failures):
            html += '<td class="cell_error"> <input class="toggle_fail" type="button" ' \
                    'value="%d" style="width:100%%; height:100%%"></td>' % len(res.failures)
        else:
            html += '<td class="label_ok"> 0 </td>'

        if len(res.errors):
            html += '<td class="cell_error"> <input class="toggle_error" type="button" ' \
                    'value="%d" style="width:100%%; height:100%%"></td>' % len(res.errors)
        else:
            html += '<td class="label_ok"> 0 </td>'
        html += '</tr>\n'

        if args.systematic and args.graph is not None:
            if os.path.exists(args.graph):
                # Do not put the img tag of the file, else it gets pre-loaded and
                # this can crash the browser when aggregating many tests.
                # It will only be added when clicking on the transform cell.
                html += '<tr class="graph" style="display: none"><td colspan=13></td></tr>'

    if len(res.errors):
        tmp = "%s\n\nERRORS:\n\n" % sub
        for t, s in res.errors:
            tid = t.id()
            tid1 = tid.split('(')[0].split('.')[-1]
            tid0, tid2 = tid.split('.' + tid1)
            tmp += "=" * 70 + "\n" + '%s %s [%s]:\n' % (tid1, tid2, tid0)
            tmp += "-" * 70 + "\n" + s + "\n"
        info += tmp
        if args.html is not None:
            html += '<tr class="errors" style="display: none; text-align: left; font-family: monospace">' \
                    '<td colspan=13><pre>%s</pre></td></tr>' % tmp

    if len(res.failures):
        tmp = "%s\n\nFAILURES:\n\n" % sub
        for t, s in res.failures:
            tid = t.id()
            tid1 = tid.split('(')[0].split('.')[-1]
            tid0, tid2 = tid.split('.' + tid1)
            tmp += "=" * 70 + "\n" + '%s %s [%s]:\n' % (tid1, tid2, tid0)
            tmp += "-" * 70 + "\n" + s + "\n"
        info += tmp
        if args.html is not None:
            html += '<tr class="failures" style="display: none; text-align: left; font-family: monospace">' \
                    '<td colspan=13><pre>%s</pre></td></tr>' % tmp

    if args.mailto_fail is not None and (nb_err_fail > 0) or args.mailto is not None:
        import smtplib
        try:
            from email.message import EmailMessage

            msg = EmailMessage()
            msg['From'] = '"pyvkfft" <%s>' % args.mailto if args.mailto is not None else args.mailto_fail
            msg['To'] = msg['From']
            msg['Subject'] = '[fail=%d error=%d] %s' % \
                             (len(res.failures), len(res.errors), sub)
            print("Mailing results:\nFrom: %s\nTo: \n %sSubject: %s" % (msg['to'], msg['to'], msg['Subject']))
            msg.set_content(info)

            s = smtplib.SMTP(args.mailto_smtp)
            s.send_message(msg)
            s.quit()
        except (ConnectionRefusedError, smtplib.SMTPConnectError):
            print("Could not connect to SMTP server (%s) to send email." % args.mailto_smtp)

    if args.html is not None:
        if len(args.html):
            html_out = args.html[0]
        else:
            html_out = name_next_file("pyvkfft-test%03d.html")
        with open(html_out, 'w') as f:
            f.write(html)
    sys.exit(int(nb_err_fail > 0))


if __name__ == '__main__':
    main()
