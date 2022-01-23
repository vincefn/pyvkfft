# -*- coding: utf-8 -*-

# PyVkFFT
#   (c) 2022- : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr
#
#
# script to run a long multi-test accuracy suite

import os


def main():
    gpu = 'v100'
    nproc0 = 20
    gpu_gb = 32
    dry_run = False
    backend = 'cupy'

    # Basic test
    com = "pyvkfft-test --nproc %d --html --range-mb 0 4100" % nproc0
    if dry_run:
        print(com)
    else:
        os.system(com)

    # systematic tests
    vtransform = ['        ', ' --r2c  ', ' --dct 1', ' --dct 2', ' --dct 3', ' --dct 4']
    # vtransform = ['        ', ' --r2c  ']
    vdim = [1, 2, 3]
    vnorm = [' --norm 1', ' --norm 0']
    vlut = ['', ' --lut']
    vprec = ['', ' --double']
    vradix = [' --radix', ' --bluestein']
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
                                nproc1 = gpu_gb // (mem / 1024 ** 3 * 1.5)
                                nproc = max(1, min(nproc1, nproc0))
                                com = 'pyvkfft-test --systematic --backend %s --gpu %s --graph --html' % (backend, gpu)
                                com += ' --max-nb-tests 0'
                                com += ' --nproc %2d --ndim %d --range %d %6d' % (nproc, ndim, n1, n2)
                                com += transform + radix + inplace + prec + lut + norm + ' --range-mb 0 4100'
                                if dry_run:
                                    print(com)
                                else:
                                    os.system(com)

    # Last, run a few 2D and 3D tests where the lengths can differ,
    # and radix and Bluestein transforms are mixed.
    for transform in ['', ' --r2c']:
        norm = ' --norm 1'
        for radix, lut, inplace, prec, ndim, n1, n2, rn in \
                [(' --radix', '', '', '', 2, 2, 4500, ' --range-nd-narrow 0.02 4 --radix-max-pow 3'),
                 (' --radix', ' --lut', '', '', 2, 2, 4500, ' --range-nd-narrow 0.02 4 --radix-max-pow 3'),
                 (' --radix', '', ' --inplace', '', 2, 2, 4500, ' --range-nd-narrow 0.02 4 --radix-max-pow 3'),
                 (' --radix', '', '', ' --double', 2, 2, 4500, ' --range-nd-narrow 0.02 4 --radix-max-pow 3'),
                 (' --radix', '', '', '', 3, 2, 150, ' --range-nd-narrow 0.02 4 --radix-max-pow 3'),
                 (' --radix', ' --lut', '', '', 3, 2, 150, ' --range-nd-narrow 0.02 4 --radix-max-pow 3'),
                 (' --radix', '', ' --inplace', '', 3, 2, 150, ' --range-nd-narrow 0.02 4 --radix-max-pow 3'),
                 (' --radix', '', '', ' --double', 3, 2, 150, ' --range-nd-narrow 0.02 4 --radix-max-pow 3'),
                 ('', '', '', '', 2, 1008, 1040, ' --range-nd-narrow 0.02 4'),
                 ('', '', '', '', 2, 2032, 2064, ' --range-nd-narrow 0.02 4'),
                 ('', '', '', '', 2, 4080, 4112, ' --range-nd-narrow 0.02 4'),
                 ('', '', '', '', 3, 120, 140, ' --range-nd-narrow 0.02 4'),
                 ]:
            mem = n2 ** ndim * 8
            if 'double' in prec:
                mem *= 2
            if 'inplace' not in inplace:
                mem *= 2
            if 'dct' in transform or 'r2c' in transform:
                mem /= 2
            nproc1 = gpu_gb // (mem / 1024 ** 3 * 1.5)
            nproc = max(1, min(nproc1, nproc0))
            com = 'pyvkfft-test --systematic --backend %s --gpu %s --graph --html' % (backend, gpu)
            com += ' --max-nb-tests 0'
            com += ' --nproc %2d --ndim %d --range %d %6d' % (nproc, ndim, n1, n2)
            com += transform + radix + inplace + prec + lut + norm + rn + ' --range-mb 0 4100'

            if dry_run:
                print(com)
                os.system(com + ' --dry-run')
            else:
                os.system(com)


if __name__ == '__main__':
    main()
