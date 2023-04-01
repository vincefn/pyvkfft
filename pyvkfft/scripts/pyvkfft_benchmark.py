# -*- coding: utf-8 -*-

# PyVkFFT
#   (c) 2022- : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr
#
#
# pyvkfft script to run a standardised benchmark


import argparse
import time
from datetime import datetime
import socket
import sqlite3
from pyvkfft.benchmark import test_gpyfft, test_skcuda, test_pyvkfft_opencl, test_pyvkfft_cuda, \
    bench_gpyfft, bench_skcuda, bench_pyvkfft_cuda, bench_pyvkfft_opencl
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


def run_test(config, gpu_name, inplace: bool = True, precision: str = 'single', language='cuda', lib='pyvkfft',
             opencl_platform=None, verbose=False, db=None, compare=False):
    # results = []
    dbc = None
    dbc0 = None
    first = True
    for c in config:
        c.precision = precision
        c.inplace = inplace
        sh = c.shape
        ndim = c.ndim
        nb_repeat = 5
        if language == 'cuda':
            if lib == 'pyvkfft':
                dt, gbps, gpu_name_real = bench_pyvkfft_cuda(sh, precision, ndim, nb_repeat, gpu_name)
            elif lib == 'skcuda':
                dt, gbps, gpu_name_real = bench_skcuda(sh, precision, ndim, nb_repeat, gpu_name)
            else:
                raise RuntimeError(f"Unknown library to benchmark:{lib}")
        else:
            if lib == 'pyvkfft':
                dt, gbps, gpu_name_real = bench_pyvkfft_opencl(sh, precision, ndim, nb_repeat, gpu_name,
                                                               opencl_platform=opencl_platform)
            elif lib == 'skcuda':
                dt, gbps, gpu_name_real = bench_gpyfft(sh, precision, ndim, nb_repeat, gpu_name,
                                                       opencl_platform=opencl_platform)
            else:
                raise RuntimeError(f"Unknown library to benchmark:{lib}")
        # results.append({'transform': str(c), 'gbps': gbps, 'dt': dt, 'gpu': gpu_name_real})
        g = gpu_name_real.replace(' ', '_').replace(':', '_')
        if db:
            if first:
                if type(db) != str:
                    db = f"pyvkfft{__version__}-{vkfft_version()}-" \
                         f"{g}-{language}-" \
                         f"{datetime.now().strftime('%Y_%m_%d_%Hh_%Mm_%Ss')}-benchmark.sql"

                hostname = socket.gethostname()
                db = sqlite3.connect(db)
                dbc = db.cursor()
                dbc.execute('CREATE TABLE IF NOT EXISTS pyvkfft_benchmark (epoch int, hostname text,'
                            'library text, language text, transform text, shape text,'
                            'ndim int, precision text, inplace int, gbps float, gpu text)')
                db.commit()
            dbc.execute('INSERT INTO pyvkfft_benchmark VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                        (time.time(), hostname, lib, language, c.transform,
                         'x'.join(str(i) for i in sh), ndim, precision, inplace, gbps, g))
            db.commit()
        if compare and first:
            dbc0 = sqlite3.connect(compare).cursor()
        if verbose:
            s = f"{str(c):>30} {gbps:6.1f} GB/s {gpu_name_real} {language:6^} "
            if db is not None and compare:
                # Find similar result
                q = f"SELECT * from pyvkfft_benchmark WHERE library = '{lib}' " \
                    f"AND gpu = '{g}' AND language = '{language}' AND transform = '{c.transform}'" \
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
    parser.add_argument('--language', action='store', choices=['cuda', 'opencl'],
                        default='cuda', help="GPU language to test")
    parser.add_argument('--library', action='store', choices=['pyvkfft', 'gpyfft', 'skcuda'],
                        default='pyvkfft', help="GPU FFT library to test")
    parser.add_argument('--precision', action='store', choices=['single', 'double'],
                        default='single', help="Precision for the benchmark")
    parser.add_argument('--gpu', action='store', type=str, default=None, help="GPU name (or sub-string)")
    parser.add_argument('--verbose', action='store_true', help="Verbose ?")
    parser.add_argument('--save', action='store_true', default=False, help="Save results to an sql file")
    parser.add_argument('--compare', action='store', type=str,
                        help="Name of database file to compare to.")
    args = parser.parse_args()
    run_test(default_config, args.gpu, language=args.language, verbose=args.verbose,
             db=args.save, compare=args.compare)


if __name__ == '__main__':
    main()
