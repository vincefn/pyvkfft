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
from pyvkfft.benchmark import test_gpyfft, test_skcuda, test_pyvkfft_opencl, test_pyvkfft_cuda, test_cupy, \
    bench_gpyfft, bench_skcuda, bench_pyvkfft_cuda, bench_pyvkfft_opencl, bench_cupy
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


def run_test(config, args):
    # results = []
    gpu_name = args.gpu
    inplace = True
    precision = args.precision
    backend = args.backend
    opencl_platform = None
    verbose = args.verbose
    db = args.save
    compare = args.compare
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
            dt, gbps, gpu_name_real = bench_pyvkfft_cuda(sh, precision, ndim, nb_repeat, gpu_name, args=vars(args))
        elif backend == 'opencl':
            dt, gbps, gpu_name_real = bench_pyvkfft_opencl(sh, precision, ndim, nb_repeat, gpu_name,
                                                           opencl_platform=opencl_platform, args=vars(args))
        elif backend == 'skcuda':
            dt, gbps, gpu_name_real = bench_skcuda(sh, precision, ndim, nb_repeat, gpu_name)
        elif backend == 'gpyfft':
            dt, gbps, gpu_name_real = bench_gpyfft(sh, precision, ndim, nb_repeat, gpu_name,
                                                   opencl_platform=opencl_platform)
        elif backend == 'cupy':
            dt, gbps, gpu_name_real = bench_cupy(sh, precision, ndim, nb_repeat, gpu_name)
        if gpu_name_real is None or gbps == 0:
            # Something went wrong ? Possible timeout ?
            continue
        # results.append({'transform': str(c), 'gbps': gbps, 'dt': dt, 'gpu': gpu_name_real})
        g = gpu_name_real.replace('Apple', '')
        g = g.strip(' _').replace(' ', '_').replace(':', '_')
        if db:
            if first:
                if type(db) != str:
                    db = f"pyvkfft{__version__}-{vkfft_version()}-" \
                         f"{g}-{backend}-" \
                         f"{datetime.now().strftime('%Y_%m_%d_%Hh_%Mm_%Ss')}-benchmark.sql"

                hostname = socket.gethostname()
                db = sqlite3.connect(db)
                dbc = db.cursor()
                dbc.execute('CREATE TABLE IF NOT EXISTS config (epoch int, hostname text,'
                            'pyvkfft text, vkfft text, backend text, transform text,'
                            'precision text, inplace int, gpu text, disableReorderFourStep int,'
                            'coalescedMemory int, numSharedBanks int,'
                            'aimThreads int, performBandwidthBoost int, registerBoost int,'
                            'registerBoostNonPow2 int, registerBoost4Step int, warpSize int, useLUT int,'
                            'batchedGroup text)')
                dbc.execute('INSERT INTO config VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                            (time.time(), hostname, __version__, vkfft_version(), backend, c.transform,
                             precision, inplace, g, args.disableReorderFourStep,
                             args.coalescedMemory, args.numSharedBanks,
                             args.aimThreads, args.performBandwidthBoost, args.registerBoost,
                             args.registerBoostNonPow2, args.registerBoost4Step, args.warpSize,
                             args.useLUT, 'x'.join(str(i) for i in args.batchedGroup)))
                db.commit()
                dbc.execute('CREATE TABLE IF NOT EXISTS benchmark (epoch int, ndim int, shape text, gbps float)')
                db.commit()
            dbc.execute('INSERT INTO benchmark VALUES (?,?,?,?)',
                        (time.time(), ndim, 'x'.join(str(i) for i in sh), gbps))
            db.commit()
        if compare and first:
            dbc0 = sqlite3.connect(compare).cursor()
        if verbose:
            s = f"{str(c):>30} {gbps:6.1f} GB/s {gpu_name_real} {backend:6^} "
            if compare:
                # Find similar result
                q = f"SELECT * from benchmark WHERE shape = '{'x'.join(str(i) for i in sh)}' ORDER by epoch"
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
    epilog = "Examples:\n" \
             "* Simple benchmark for radix transforms:\n" \
             "     pyvkfft-benchmark --backend cuda --gpu titan --verbose\n\n" \
             "* Systematic benchmark for 1D radix transforms over a given range:\n" \
             "     pyvkfft-benchmark --backend cuda --gpu titan --systematic --ndim 1 --range 2 256 --verbose\n\n" \
             "* Same but only for powers of 2 and 3 sizes, in 2D, and save the results " \
             "to an SQL file for later plotting:\n" \
             "     pyvkfft-benchmark --backend cuda --gpu titan --systematic --radix 2 3 " \
             "--ndim 2 --range 2 256 --verbose --save\n\n" \
             "* plot the result of a benchmark:\n" \
             "     pyvkfft-benchmark --plot pyvkfft-version-gpu-date-etc.sql\n\n" \
             "* plot & compare the results of multiple benchmarks (grouped by 1D/2D/3D transforms):\n" \
             "     pyvkfft-benchmark --plot *.sql\n\n"

    desc = "Run pyvkfft benchmark tests. This is pretty slow as each test runs " \
           "in a separate process (including the GPU initialisation) - this is done to avoid " \
           "any context a memory issues when performing a large number of tests. " \
           "This can also be used to compare results with cufft (scikit-cuda or cupy) and gpyfft. " \
           ""

    parser = argparse.ArgumentParser(prog='pyvkfft-benchmark', epilog=epilog,
                                     description=desc,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--backend', action='store', choices=['cuda', 'opencl', 'gpyfft', 'skcuda', 'cupy'],
                        default='pyvkfft', help="FFT backend to use, 'cuda' and 'opencl' will "
                                                "use pyvkfft with the corresponding language.")
    parser.add_argument('--precision', action='store', choices=['single', 'double'],
                        default='single', help="Precision for the benchmark")
    parser.add_argument('--gpu', action='store', type=str, default=None, help="GPU name (or sub-string)")
    parser.add_argument('--opencl_platform', action='store',
                        help="Name (or sub-string) of the opencl platform to use (case-insensitive). "
                             "Note that by default the PoCL platform is skipped, "
                             "unless it is specifically requested or it is the only one available "
                             "(PoCL has some issues with VkFFT for some transforms)")
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
                             "for comparison. This parameter supersedes all others (no tests "
                             "are run if --plot is given)")
    sysgrp = parser.add_argument_group("systematic", "Options for --systematic:")
    sysgrp.add_argument('--radix', action='store', nargs='+', type=int,
                        help="Perform only radix transforms. By default, all available "
                             "radix transforms are allowed. Alternatively a list can be given: "
                             "'--radix 2' (only 2**n array sizes), '--radix 2 3 5' "
                             "(only 2**N1 * 3**N2 * 5**N3)",
                        choices=[2, 3, 5, 7, 11, 13], default=[2, 3, 5, 7, 11, 13])
    sysgrp.add_argument('--bluestein', '--rader', action='store_true', default=False,
                        help="Test only non-radix sizes, using the Bluestein or Rader transforms. "
                             "Not compatible with --radix")
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
                        default=[0, 128])
    sysgrp = parser.add_argument_group("advanced", "Advanced options for VkFFT. Do NOT use unless you "
                                                   "really know what these mean. -1 will always "
                                                   "defer the choice to VkFFT.")
    sysgrp.add_argument('--disableReorderFourStep', action='store', choices=[-1, 0, 1], type=int,
                        default=-1, help="Disables unshuffling of Four step algorithm."
                                         " Requires tempbuffer allocation")
    sysgrp.add_argument('--coalescedMemory', action='store', choices=[-1, 16, 32, 64, 128], type=int,
                        default=-1, help="Number of bytes to coalesce per one transaction: "
                                         "defaults to 32 for Nvidia and AMD, 64 for others."
                                         "Should be a power of two")
    sysgrp.add_argument('--numSharedBanks', action='store', choices=[-1] + list(range(16, 64 + 1, 4)), type=int,
                        default=-1, help="Number of shared banks on the target GPU. Default is 32. ")
    sysgrp.add_argument('--aimThreads', action='store', choices=[-1] + list(range(16, 256 + 1, 4)), type=int,
                        default=-1, help="Try to aim all kernels at this amount of threads. ")
    sysgrp.add_argument('--performBandwidthBoost', action='store', choices=[-1, 0, 1, 2, 4], type=int,
                        default=-1, help="Try to reduce coalesced number by a factor of X"
                                         "to get bigger sequence in one upload for strided axes. ")
    sysgrp.add_argument('--registerBoost', action='store', choices=[-1, 1, 2, 4], type=int,
                        default=-1, help="Specify if the register file size is bigger than "
                                         "shared memory and can be used to extend it X times "
                                         "(on Nvidia 256KB register  file can be used instead "
                                         "of 32KB of shared memory, set this constant to 4 to "
                                         "emulate 128KB of shared memory). ")
    sysgrp.add_argument('--registerBoostNonPow2', action='store', choices=[-1, 0, 1], type=int,
                        default=-1, help="Specify if register over-utilization should "
                                         "be used on non-power of 2 sequences ")
    sysgrp.add_argument('--registerBoost4Step', action='store', choices=[-1, 1, 2, 4], type=int,
                        default=-1, help="Specify if register file over-utilization "
                                         "should be used in big sequences (>2^14), "
                                         "same definition as registerBoost ")
    sysgrp.add_argument('--warpSize', action='store', choices=[-1, 1, 2, 4, 8, 16, 32, 64, 128, 256], type=int,
                        default=-1, help="Number of threads per warp/wavefront. Normally automatically "
                                         "derived from the driver. Must be a power of two")
    sysgrp.add_argument('--batchedGroup', action='store', nargs=3, type=int, default=[-1, -1, -1],
                        help="How many FFTs are done per single kernel "
                             "by a dedicated thread block, for each dimension.")
    sysgrp.add_argument('--useLUT', action='store', choices=[-1, 0, 1], type=int,
                        default=-1, help="Use a look-up table to bypass the native sincos functions.")
    args = parser.parse_args()
    if args.plot:
        import matplotlib.pyplot as plt
        res_all = {}
        vgpu = []
        vbackend = []
        vopt = []
        for ndim in (1, 2, 3):
            for src in args.plot:
                dbc0 = sqlite3.connect(src).cursor()

                dbc0.execute(f"SELECT * from config")
                r = dbc0.fetchone()
                config = {col[0]: r[i] for i, col in enumerate(dbc0.description)}
                gpu = config['gpu']
                if gpu not in vgpu:
                    vgpu.append(gpu)
                if config['backend'] not in vbackend:
                    vbackend.append(config['backend'])
                for k, v in {"disableReorderFourStep": "r4s", "coalescedMemory": "coalmem",
                             "numSharedBanks": "nbanks", "aimThreads": "threads",
                             "performBandwidthBoost": "bwboost", "registerBoost": "rboost",
                             "registerBoostNonPow2": "rboostn2", "registerBoost4Step": "rboost4",
                             "warpSize": "warp", "useLUT": "lut", "batchedGroup": "batch"}.items():
                    if k in config:
                        if k == "batchedGroup":
                            # config[k] = [int(b) for b in v.split('x')]
                            # print(config[k])
                            if config[k] != '-1x-1x-1' and v not in vopt:
                                vopt.append(v)
                        elif config[k] != -1 and v not in vopt:
                            vopt.append(v)
                vkfft_ver = config['vkfft']

                dbc0.execute(f"SELECT * from benchmark WHERE ndim = {ndim} ORDER by epoch")
                res = dbc0.fetchall()
                if len(res):
                    vk = [k[0] for k in dbc0.description]
                    igbps = vk.index('gbps')
                    vgbps = [r[igbps] for r in res]
                    ish = vk.index('shape')
                    vlength = [int(r[ish].split('x')[-1]) for r in res]

                    if config['backend'] in ['skcuda', 'cupy', 'gpyfft']:
                        k = f"{config['backend']}[{gpu}]"
                    else:
                        k = f"VkFFT.{config['backend']} {vkfft_ver}[{gpu}]"
                        if config['warpSize'] != -1:
                            k += f"-warp{config['warpSize']}"
                        if config['registerBoost'] != -1:
                            k += f"-rboost{config['registerBoost']}"
                        if config['registerBoostNonPow2'] != -1:
                            k += f"-rboostn2{config['registerBoostNonPow2']}"
                        if config['coalescedMemory'] != -1:
                            k += f"-coalmem{config['coalescedMemory']}"
                        if config['aimThreads'] != -1:
                            k += f"-threads{config['aimThreads']}"
                        if config['numSharedBanks'] != -1:
                            k += f"-banks{config['numSharedBanks']}"
                        if 'batchedGroup' in config:
                            if config['batchedGroup'] != '-1x-1x-1':
                                k += f"-batch{config['batchedGroup']}"

                    if ndim not in res_all:
                        res_all[ndim] = {k: [vlength, vgbps]}
                    else:
                        res_all[ndim][k] = [vlength, vgbps]
        vgpu.sort()
        vbackend.sort()
        vopt.sort()
        str_config = "_".join(vgpu) + f"-{','.join(vbackend)}"
        if len(vopt):
            str_opt = "_".join(vopt)
        else:
            str_opt = ""
        for ndim, res in res_all.items():
            plt.figure(figsize=(16, 8))
            for k, v in res.items():
                x, y = v
                plt.plot(x, y, '.', label=k)

            plt.xlabel("array length")
            plt.ylabel("Theoretical throughput (GBytes/s)")
            plt.ylim(0)
            plt.title(f"{ndim}D FFT-" + str_config)

            # Use powers of 2 for xticks
            xmin, xmax = plt.xlim()
            step = 2 ** (round(np.log2(xmax - xmin + 1) - 4))
            xmin -= xmin % step
            plt.xticks(np.arange(xmin, xmax, step))

            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"pyvkfft-benchmark-{str_config}-{ndim}D-{str_opt}.png")
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
            if args.bluestein and args.radix != [2, 3, 5, 7, 11, 13]:
                raise RuntimeError("--bluestein cannot be used with --radix")
            vshape = np.array(radix_gen_n(nmax=args.range[1], max_size=size_min_max[1],
                                          radix=args.radix, ndim=1, even=False,
                                          nmin=args.range[0], max_pow=None,
                                          range_nd_narrow=None, min_size=size_min_max[0],
                                          inverted=args.bluestein),
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
        run_test(config, args)


if __name__ == '__main__':
    main()
