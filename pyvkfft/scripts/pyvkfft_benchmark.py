# -*- coding: utf-8 -*-

# PyVkFFT
#   (c) 2022- : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr
#
#
# pyvkfft script to run a standardised benchmark


import argparse
import numpy as np
import time
from datetime import datetime
import socket
import sqlite3
from pyvkfft.benchmark import test_gpyfft, test_skcuda, test_pyvkfft_opencl, test_pyvkfft_cuda, \
    bench_gpyfft, bench_skcuda, bench_pyvkfft_cuda, bench_pyvkfft_opencl
from pyvkfft.base import radix_gen_n
from pyvkfft.version import __version__, vkfft_version


class BenchConfig:
    def __init__(self, transform: str, shape, ndim: int, inplace: bool = True, precision: str = 'single'):
        self.transform = transform
        self.shape = shape
        self.ndim = ndim
        self.inplace = inplace
        self.precision = precision

    def __str__(self):
        return f"{self.transform}_{'x'.join([str(i) for i in self.shape])}_{self.ndim}D_" \
               f"{'i' if self.inplace else 'o'}_{'s' if self.precision == 'single' else 'double'}"


default_config = [
    BenchConfig('c2c', (100, 256), 1),
    BenchConfig('c2c', (100, 1024), 1),
    BenchConfig('c2c', (100, 10000), 1),
    BenchConfig('c2c', (10, 2 * 3 * 5 * 7 * 11 * 13), 1),  # 30030
    BenchConfig('c2c', (100, 17 * 19), 1),  # 323
    BenchConfig('c2c', (100, 2 ** 16 + 1), 1),  # 65537
    BenchConfig('c2c', (20, 256, 256), 2),
    BenchConfig('c2c', (10, 1024, 1024), 2),
    BenchConfig('c2c', (10, 2560, 2160), 2),
    BenchConfig('c2c', (4, 4200, 4200), 2),
    BenchConfig('c2c', (10, 7 * 11 * 13, 7 * 11 * 13), 2),  # 1001
    BenchConfig('c2c', (100, 17 * 19, 17 * 19), 2),  # 323
    BenchConfig('c2c', (256, 256, 256), 3),
    BenchConfig('c2c', (512, 512, 512), 3),
    # BenchConfig('r2c', (100, 1024), 1),
    # BenchConfig('r2c', (10, 2 * 3 * 5 * 7 * 11 * 13), 1),  # 30030
    # BenchConfig('r2c', (20, 256, 256), 2),
    # BenchConfig('r2c', (10, 2560, 2120), 2),
]


def run_test(config, gpu_name, inplace: bool = True, precision: str = 'single', backend='cuda',
             opencl_platform=None, verbose=False, db=None, compare=False):
    # results = []
    dbc = None
    dbc0 = None
    first = True
    for c in config:
        c.precision = precision
        c.inplace = inplace
        sh = tuple(c.shape)
        ndim = c.ndim
        nb_repeat = 5
        if backend == 'cuda':
            dt, gbps, gpu_name_real = bench_pyvkfft_cuda(sh, precision, ndim, nb_repeat, gpu_name)
        elif backend == 'opencl':
            dt, gbps, gpu_name_real = bench_pyvkfft_opencl(sh, precision, ndim, nb_repeat, gpu_name,
                                                           opencl_platform=opencl_platform)
        elif backend == 'skcuda':
            dt, gbps, gpu_name_real = bench_skcuda(sh, precision, ndim, nb_repeat, gpu_name)
        elif backend == 'skcuda':
            dt, gbps, gpu_name_real = bench_gpyfft(sh, precision, ndim, nb_repeat, gpu_name,
                                                   opencl_platform=opencl_platform)
        # results.append({'transform': str(c), 'gbps': gbps, 'dt': dt, 'gpu': gpu_name_real})
        g = gpu_name_real.replace(' ', '_').replace(':', '_')
        if db:
            if first:
                if type(db) != str:
                    db = f"pyvkfft{__version__}-{vkfft_version()}-" \
                         f"{g}-{backend}-" \
                         f"{datetime.now().strftime('%Y_%m_%d_%Hh_%Mm_%Ss')}-benchmark.sql"

                hostname = socket.gethostname()
                db = sqlite3.connect(db)
                dbc = db.cursor()
                dbc.execute('CREATE TABLE IF NOT EXISTS pyvkfft_benchmark (epoch int, hostname text,'
                            'pyvkfft text, vkfft text, backend text, transform text, shape text,'
                            'ndim int, precision text, inplace int, gbps float, gpu text)')
                db.commit()
            dbc.execute('INSERT INTO pyvkfft_benchmark VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                        (time.time(), hostname, __version__, vkfft_version(), backend, c.transform,
                         'x'.join(str(i) for i in sh), ndim, precision, inplace, gbps, g))
            db.commit()
        if compare and first:
            dbc0 = sqlite3.connect(compare).cursor()
        if verbose:
            s = f"{str(c):>30} {gbps:6.1f} GB/s {gpu_name_real} {backend:6^} "
            if compare:
                # Find similar result
                q = f"SELECT * from pyvkfft_benchmark WHERE backend = '{backend}' " \
                    f"AND gpu = '{g}' AND transform = '{c.transform}'" \
                    f"AND shape = '{'x'.join(str(i) for i in sh)}' AND ndim = {ndim} " \
                    f"AND precision = '{precision}' AND inplace = {int(inplace)} ORDER by epoch"
                dbc0.execute(q)
                res = dbc0.fetchall()
                idx = [k[0] for k in dbc0.description].index('gbps')
                if len(res):
                    r = res[-1]
                    gbps0 = r[idx]
                    s += f"  ref: {gbps / gbps0 * 100:3.0f}% [{gbps0:6.1f} GB/s]"
                    if True:  # colour_output:
                        a = max(0.5, min(gbps / gbps0, 1.5))
                        if a <= 0.9:
                            s = "\x1b[31m" + s + "\x1b[0m"
                        elif a >= 1.1:
                            s = "\x1b[32m" + s + "\x1b[0m"

            if first:
                print(f"pyvkfft: {__version__}   VkFFT: {vkfft_version()}")
                first = False

            print(s)


def main():
    parser = argparse.ArgumentParser(prog='pyvkfft-benchmark',
                                     description='Run pyvkfft benchmark tests')
    parser.add_argument('--backend', action='store', choices=['cuda', 'opencl', 'gpyfft', 'skcuda'],
                        default='pyvkfft', help="FFT backend to use, 'cuda' and 'opencl' will "
                                                "use pyvkfft with the corresponding language.")
    parser.add_argument('--precision', action='store', choices=['single', 'double'],
                        default='single', help="Precision for the benchmark")
    parser.add_argument('--gpu', action='store', type=str, default=None, help="GPU name (or sub-string)")
    parser.add_argument('--verbose', action='store_true', help="Verbose ?")
    parser.add_argument('--save', action='store_true', default=False, help="Save results to an sql file")
    parser.add_argument('--compare', action='store', type=str,
                        help="Name of database file to compare to.")
    parser.add_argument('--systematic', action='store_true',
                        help="Perform a systematic benchmark over a range of array sizes.\n"
                             "Without this argument only a small number of array sizes is tested.")
    parser.add_argument('--dry-run', action='store_true',
                        help="Perform a dry-run, printing the number of array shapes to test")
    parser.add_argument('--plot', action='store', nargs='+', type=str,
                        help="Plot results stored in *.sql files. Separate plots are given "
                             "for different dimensions. Multiple *.sql files can be given "
                             "for comparison. This parameter supersedes all others.")
    sysgrp = parser.add_argument_group("systematic", "Options for --systematic:")
    sysgrp.add_argument('--radix', action='store', nargs='*', type=int,
                        help="Perform only radix transforms. If no value is given, all available "
                             "radix transforms are allowed. Alternatively a list can be given: "
                             "'--radix 2' (only 2**n array sizes), '--radix 2 3 5' "
                             "(only 2**N1 * 3**N2 * 5**N3)",
                        choices=[2, 3, 5, 7, 11, 13], default=[2, 3, 5, 7, 11, 13])
    sysgrp.add_argument('--ndim', action='store', nargs='+',
                        help="Number of dimensions for the transform. The arrays will be "
                             "stacked so that each batch transform is at least 1GB.",
                        default=[2], type=int, choices=[1, 2, 3])
    sysgrp.add_argument('--range', action='store', nargs=2, type=int,
                        help="Range of array lengths [min, max] along each transform dimension, "
                             "'--range 2 128'. This is combined with --range-mb to determine the "
                             "actual range, so you can put large values here and let the maximum "
                             "total size limit the actual memory used.",
                        default=[2, 256])
    sysgrp.add_argument('--range-mb', action='store', nargs=2, type=int,
                        help="Range of array sizes in MBytes. This is combined with --range to"
                             "find the actual range to use.",
                        default=[2, 128])
    args = parser.parse_args()
    if args.plot:
        import matplotlib.pyplot as plt
        res_all = {}
        for ndim in (1, 2, 3):
            for src in args.plot:
                dbc0 = sqlite3.connect(src).cursor()
                dbc0.execute(f"SELECT * from pyvkfft_benchmark WHERE ndim = {ndim} ORDER by epoch")
                res = dbc0.fetchall()
                if len(res):
                    vk = [k[0] for k in dbc0.description]
                    gpu = res[0][vk.index('gpu')]
                    vkfft_ver = res[0][vk.index('vkfft')]

                    igbps = vk.index('gbps')
                    vgbps = [r[igbps] for r in res]
                    ish = vk.index('shape')
                    vlength = [int(r[ish].split('x')[-1]) for r in res]

                    if ndim not in res_all:
                        res_all[ndim] = {f"VkFFT {vkfft_ver}[{gpu}]": [vlength, vgbps]}
                    else:
                        res_all[ndim][f"VkFFT {vkfft_ver}[{gpu}]"] = [vlength, vgbps]
        for ndim, res in res_all.items():
            plt.figure(figsize=(16, 8))
            for k, v in res.items():
                x, y = v
                plt.plot(x, y, '.', label=k)

            plt.xlabel("array length")
            plt.ylabel("Theoretical throughput (GBytes/s)")
            plt.ylim(0)

            # Use powers of 2 for xticks
            xmin, xmax = plt.xlim()
            step = 2 ** (round(np.log2(xmax - xmin + 1) - 4))
            xmin -= xmin % step
            plt.xticks(np.arange(xmin, xmax, step))

            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"pyvkfft-benchmark-{ndim}D.png")
        return
    if args.systematic:
        config = []
        for ndim in args.ndim:
            size_min_max = np.array(args.range_mb) * 1024 ** 2
            if args.precision == 'double':
                size_min_max //= 16
            else:
                size_min_max //= 8
            size_min_max = np.round(size_min_max ** (1 / ndim)).astype(int)
            vshape = np.array(radix_gen_n(nmax=args.range[1], max_size=size_min_max[1],
                                          radix=args.radix, ndim=1, even=False,
                                          nmin=args.range[0], max_pow=None,
                                          range_nd_narrow=None, min_size=size_min_max[0]),
                              dtype=int).flatten()
            nbatch = 1e8 / (vshape ** ndim * (8 if args.precision == 'double' else 4))
            nbatch = np.maximum(1, nbatch).astype(int)
            config += [BenchConfig('c2c', [b] + [n] * ndim, ndim) for b, n in zip(nbatch, vshape)]
    else:
        config = default_config
    if args.dry_run:
        for c in config:
            print(c)
        print("Total number of arrays to test: ", len(config))
    else:
        run_test(config, args.gpu, backend=args.backend, verbose=args.verbose,
                 db=args.save, compare=args.compare)


if __name__ == '__main__':
    main()
