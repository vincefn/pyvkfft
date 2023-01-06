# -*- coding: utf-8 -*-

# PyVkFFT
#   (c) 2022- : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr
#
#
# script to run a long multi-test accuracy suite

import os
import argparse
import sys


def make_parser():
    epilog = "Examples:\n" \
             "   pyvkfft-test-suite --gpumem 32 --backend pycuda --gpu V100\n" \
             "This will run the full test suite with a maximum of 20 parallel\n" \
             "process (based on the available memory), using pycuda on a V100 GPU\n\n" \
             "   pyvkfft-test-suite --gpumem 32 --backend pycuda --gpu V100 --transform c2c --ndim 1 2\n" \
             "This will run the test suite only for 1D and 2D C2C transforms\n" \
             "WARNING: due to the level of multi-processing used, this suite is \n" \
             "very difficult to interrupt !"
    parser = argparse.ArgumentParser(prog='pyvkfft-test-suite', epilog=epilog,
                                     description='Run pyvkfft unit tests, regular or systematic',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gpu', action='store',
                        help="Name (or sub-string) of the GPU to use. If not given, "
                             "the first available will be used")
    parser.add_argument('--nproc', action='store', nargs=1,
                        help="Maximum number of parallel process to use to speed up tests. "
                             "This number will be decreased for larger arrays (e.g. 3D), but"
                             "it should be checked to avoid memory errors. A good value"
                             "for 32 GB is 20 processes.",
                        default=10, type=int)
    parser.add_argument('--gpumem', action='store', nargs=1,
                        help="Available GPU memory.",
                        default=16, type=int)
    parser.add_argument('--backend', action='store', required=True,
                        help="GPU backend",
                        choices=['pycuda', 'cupy', 'pyopencl'])
    parser.add_argument('--opencl_platform', action='store', default=None,
                        help="Name (or sub-string) of the opencl platform to use (case-insensitive)")
    parser.add_argument('--transform', action='store', nargs='+',
                        help="Transforms to test (defaults to all)",
                        default=['c2c', 'r2c', 'dct'], choices=['c2c', 'r2c', 'dct'])
    parser.add_argument('--radix', action='store_true',
                        help="Use this option to only test radix transforms")
    parser.add_argument('--ndim', action='store', nargs='+',
                        help="Number of dimensions for the tests. Several values can be given, "
                             "e.g. 1 2 3",
                        default=[1, 2, 3], type=int, choices=[1, 2, 3])
    parser.add_argument('--dry-run', action='store_true',
                        help="Perform a dry-run, printing the number of tests")
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    backend = args.backend
    dry_run = args.dry_run
    gpu = args.gpu
    gpumem = args.gpumem
    nproc0 = args.nproc
    opencl_platform = args.opencl_platform
    radix = args.radix
    vdim = args.ndim

    # Basic test
    com = "pyvkfft-test --nproc %d --html --range-mb 0 4100" % nproc0
    if opencl_platform is not None:
        com += ' --opencl_platform ' + opencl_platform
    if dry_run:
        print(com)
    else:
        if os.system(com) == 2:
            # Keyboard interrupt (why 2 ?)
            print("Aborting test suite")
            sys.exit(1)

    # systematic tests
    vtransform = []
    if 'c2c' in args.transform:
        vtransform.append('        ')
    if 'r2c' in args.transform:
        vtransform.append(' --r2c  ')
    if 'dct' in args.transform:
        for i in range(1, 4 + 1):
            vtransform.append(' --dct %d' % i)
    vnorm = [' --norm 1', ' --norm 0']
    vlut = ['', ' --lut']
    vprec = ['', ' --double']
    vradix = [' --radix'] if radix else [' --radix', ' --bluestein']
    vinplace = ['', ' --inplace']
    for radix in vradix:
        for norm in vnorm:
            for lut in vlut:
                for inplace in vinplace:
                    for prec in vprec:
                        if ' --lut' in lut and 'double' in prec:
                            continue
                        for transform in vtransform:
                            if 'dct' in transform and '0' in norm:
                                continue
                            for ndim in vdim:
                                n1 = 3 if 'dct 4' in transform else 2
                                if ndim == 1:
                                    if 'dct 1' in transform:
                                        n2 = 767 if 'double' in prec else 1535
                                    elif 'dct' in transform:
                                        if 'double' in prec:
                                            n2 = 1536 if 'bluestein' in radix else 3071
                                        else:
                                            n2 = 3071 if 'bluestein' in radix else 4096
                                    else:
                                        n2 = 100000 if 'radix' in radix else 10000
                                elif ndim == 2:
                                    if 'dct 1' in transform:
                                        n2 = 512 if 'double' in prec else 1024
                                    elif 'dct' in transform:
                                        n2 = 1024 if 'bluestein' in radix and 'double' in prec else 2047
                                    else:
                                        n2 = 4500
                                else:  # ndim==3
                                    if 'dct' in transform:
                                        n2 = 500
                                    else:
                                        n2 = 550
                                mem = n2 ** ndim * 8
                                if 'double' in prec:
                                    mem *= 2
                                if 'inplace' not in inplace:
                                    mem *= 2
                                if 'dct' in transform or 'r2c' in transform:
                                    mem /= 2
                                nproc1 = gpumem // (mem / 1024 ** 3 * 1.5)
                                nproc = max(1, min(nproc1, nproc0))
                                com = 'pyvkfft-test --systematic --backend %s' % backend
                                if gpu is not None:
                                    com += ' --gpu %s' % gpu
                                com += ' --graph --html --max-nb-tests 0'
                                com += ' --nproc %2d --ndim %d --range %d %6d' % (nproc, ndim, n1, n2)
                                com += transform + radix + inplace + prec + lut + norm + ' --range-mb 0 4100'
                                if opencl_platform is not None:
                                    com += ' --opencl_platform ' + opencl_platform
                                if dry_run:
                                    print(com)
                                    # os.system(com + ' --dry-run')
                                else:
                                    if os.system(com) == 2:
                                        # Keyboard interrupt (why 2 ?)
                                        print("Aborting test suite")
                                        sys.exit(1)

    # Last, run a few 2D and 3D tests where the lengths can differ,
    # and radix and Bluestein transforms are mixed.
    v = []
    if 2 in vdim:
        v += [(' --radix', '', '', '', 2, 2, 4500, ' --range-nd-narrow 0.02 4 --radix-max-pow 3'),
              (' --radix', ' --lut', '', '', 2, 2, 4500, ' --range-nd-narrow 0.02 4 --radix-max-pow 3'),
              (' --radix', '', ' --inplace', '', 2, 2, 4500, ' --range-nd-narrow 0.02 4 --radix-max-pow 3'),
              (' --radix', '', '', ' --double', 2, 2, 4500, ' --range-nd-narrow 0.02 4 --radix-max-pow 3')]
        if not radix:
            v += [('', '', '', '', 2, 1008, 1040, ' --range-nd-narrow 0.02 4'),
                  ('', '', '', '', 2, 2032, 2064, ' --range-nd-narrow 0.02 4'),
                  ('', '', '', '', 2, 4080, 4112, ' --range-nd-narrow 0.02 4'),
                  ]
    if 3 in vdim:
        v += [(' --radix', '', '', '', 3, 2, 150, ' --range-nd-narrow 0.02 4 --radix-max-pow 3'),
              (' --radix', ' --lut', '', '', 3, 2, 150, ' --range-nd-narrow 0.02 4 --radix-max-pow 3'),
              (' --radix', '', ' --inplace', '', 3, 2, 150, ' --range-nd-narrow 0.02 4 --radix-max-pow 3'),
              (' --radix', '', '', ' --double', 3, 2, 150, ' --range-nd-narrow 0.02 4 --radix-max-pow 3')]
        if not radix:
            v += [('', '', '', '', 3, 120, 140, ' --range-nd-narrow 0.02 4'),
                  ]

    for transform in vtransform:
        if 'dct' in transform:
            continue
        norm = ' --norm 1'
        for radix, lut, inplace, prec, ndim, n1, n2, rn in v:
            mem = n2 ** ndim * 8
            if 'double' in prec:
                mem *= 2
            if 'inplace' not in inplace:
                mem *= 2
            if 'dct' in transform or 'r2c' in transform:
                mem /= 2
            nproc1 = gpumem // (mem / 1024 ** 3 * 1.5)
            nproc = max(1, min(nproc1, nproc0))
            com = 'pyvkfft-test --systematic --backend %s' % backend
            if gpu is not None:
                com += ' --gpu %s' % gpu
            com += ' --graph --html --max-nb-tests 0'
            com += ' --nproc %2d --ndim %d --range %d %6d' % (nproc, ndim, n1, n2)
            com += transform + radix + inplace + prec + lut + norm + rn + ' --range-mb 0 4100'
            if opencl_platform is not None:
                com += ' --opencl_platform ' + opencl_platform

            if dry_run:
                print(com)
                # os.system(com + ' --dry-run')
            else:
                if os.system(com) == 2:
                    # Keyboard interrupt (why 2 ?)
                    print("Aborting test suite")
                    sys.exit(1)


if __name__ == '__main__':
    main()
