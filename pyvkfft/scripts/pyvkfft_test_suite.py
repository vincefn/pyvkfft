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
    gpu = 'p6000'
    nproc0 = 8
    gpu_gb = 11
    dry_run = True
    backend = 'pycuda'

    # Basic test
    com = "pyvkfft-test --nproc %d --html" % nproc0
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
                                        n2 = 3071 if 'double' in prec or 'bluestein' in radix else 4096
                                    else:
                                        n2 = 100000 if 'radix' in radix else 10000
                                elif ndim == 2:
                                    if 'dct 1' in transform:
                                        n2 = 512 if 'double' in prec else 1024
                                    elif 'dct' in transform:
                                        n2 = 2047
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
                                com += transform + radix + inplace + prec + lut + norm
                                if dry_run:
                                    print(com)
                                else:
                                    os.system(com)


if __name__ == '__main__':
    main()
