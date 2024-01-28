# -*- coding: utf-8 -*-

# PyVkFFT
#   (c) 2022- : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr
#
#
# pyvkfft script to run a standardised benchmark


import argparse
from string import capwords
import numpy as np
import time
from datetime import datetime
import socket
import sqlite3
from pyvkfft.benchmark import test_gpyfft, test_skcuda, test_pyvkfft_opencl, test_pyvkfft_cuda, test_cupy, \
    bench_gpyfft, bench_skcuda, bench_pyvkfft_cuda, bench_pyvkfft_opencl, bench_cupy
from pyvkfft.base import radix_gen_n, primes
from pyvkfft.version import __version__, vkfft_version


class BenchConfig:
    def __init__(self, transform: str, shape, ndim: int, inplace: bool = True, precision: str = 'single',
                 nb_loop=1):
        self.transform = transform
        self.shape = shape
        self.ndim = ndim
        self.inplace = inplace
        self.precision = precision
        self.nb_loop = nb_loop

    def __str__(self):
        return f"{self.transform}_{'x'.join([str(i) for i in self.shape])}_{self.ndim}D_" \
               f"{'i' if self.inplace else 'o'}_{'s' if self.precision == 'single' else 'd'}" \
               f" [nloop={self.nb_loop}]"


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


def plot_benchmark(*sql_files):
    import matplotlib.pyplot as plt
    from matplotlib.markers import MarkerStyle
    res_all = {}
    vgpu = []
    vbackend = []
    vopt = []
    for ndim in (1, 2, 3):
        for src in sql_files:
            dbc0 = sqlite3.connect(src).cursor()

            dbc0.execute(f"SELECT * from config")
            r = dbc0.fetchone()
            config = {col[0]: r[i] for i, col in enumerate(dbc0.description)}
            gpu = config['gpu']
            clplat = config['platform']
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

                ialgo = vk.index('algo')
                valgo = [r[ialgo] for r in res]

                iup = vk.index('nb_upload')
                vup = [r[iup] for r in res]

                ish = vk.index('shape')
                vlength = [int(r[ish].split('x')[-1]) for r in res]
                platgpu = f'{clplat}:{gpu}' if len(clplat) else gpu
                if config['backend'] in ['skcuda', 'cupy', 'gpyfft']:
                    k = f"{config['backend']}[{platgpu}]"
                else:
                    k = f"VkFFT.{config['backend']} {vkfft_ver}[{platgpu}]"
                    if config['warpSize'] != -1:
                        if config['warpSize'] == -99:
                            k += f"-warp=auto"
                        else:
                            k += f"-warp{config['warpSize']}"
                    if config['registerBoost'] != -1:
                        k += f"-rboost{config['registerBoost']}"
                    if config['registerBoostNonPow2'] != -1:
                        k += f"-rboostn2{config['registerBoostNonPow2']}"
                    if config['coalescedMemory'] != -1:
                        if config['coalescedMemory'] == -99:
                            k += f"-coalmem=auto"
                        else:
                            k += f"-coalmem{config['coalescedMemory']}"
                    if config['aimThreads'] != -1:
                        if config['aimThreads'] == -99:
                            k += f"-threads=auto"
                        else:
                            k += f"-threads{config['aimThreads']}"
                    if config['numSharedBanks'] != -1:
                        k += f"-banks{config['numSharedBanks']}"
                    if 'batchedGroup' in config:
                        if config['batchedGroup'] != '-1x-1x-1':
                            k += f"-batch{config['batchedGroup']}"
                if 'bluestein' in config['radix'].lower():
                    k += '-' + '\u0336'.join('radix') + '\u0336'
                elif 'none' not in config['radix'].lower():
                    k += f"-radix{config['radix']}"
                k += f"[{min(vlength)}-{max(vlength)}]"
                r = {'length': vlength, 'gbps': vgbps, 'backend': config['backend'],
                     'gpu': gpu, 'platform': config['platform'],
                     'algo': valgo, 'nb_upload': vup}
                if ndim not in res_all:
                    res_all[ndim] = {k: r}
                else:
                    res_all[ndim][k] = r
                print(f"{ndim}D: {src} -> {k} [{len(vlength)} entries]")
    vgpu.sort()
    vbackend.sort()
    vopt.sort()
    str_config = ",".join(vgpu) + f"-{','.join(vbackend)}"
    if len(vopt):
        str_opt = "-" + "_".join(vopt)
    else:
        str_opt = ""

    # Plot style:
    # * if multiple backends are used, one colour is used per backend
    #   and the symbol changes with the parameters
    # * If only one backend is used, the colour changes automatically with the parameters

    # Symbols used
    vsymb = ['.', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'd', 'D']
    # Colour for each backend
    vcol = {'cuda': '#FF8C00', 'opencl': '#FF00FF', 'skcuda': '#0000FF', 'cupy': '#0090FF', 'gpyfft': '#00FF00'}
    # Colours when changing parameters for 1 backend
    vcol_k = ['k', 'r', 'g', 'b', 'c', 'm', 'y', 'gray', 'chartreuse', 'maroon', 'turquoise', 'deepskyblue']
    for ndim, res in res_all.items():
        plt.figure(figsize=(16, 8))

        tmp = [v['backend'] for v in res.values()]
        if tmp.count(tmp[0]) == len(tmp):
            one_backend = True
        else:
            one_backend = False

        # Counter of results per backend
        vct = {'cuda': 0, 'opencl': 0, 'skcuda': 0, 'cupy': 0, 'gpyfft': 0}

        vk = sorted(res.keys())
        icol = 0  # iterate color when using 1 backend with multiple config
        for k in vk:
            v = res[k]
            x, y = v['length'], v['gbps']
            backend = v['backend']
            vprim = [max(primes(xx)) for xx in x]
            if backend in ['cuda', 'opencl']:
                vfill = np.array(['B' if 'B' in a else 'R' if 'R' in a else 'r' for a in v['algo']])
            elif backend in ['gpyfft']:
                vfill = ['r'] * len(vprim)
            else:
                # cufft (cupy, skcuda)
                vfill = ['B' if xx <= 7 else 'r' for xx in vprim]
            if one_backend:
                # Plot separately Bluestein, Rader and radix
                for algo in ['r', 'B', 'R']:
                    idx = np.where(vfill == algo)[0]
                    if len(idx):
                        # fillstyle={'r': 'full', 'B': 'none', 'R': 'bottom'}[algo]
                        ms = MarkerStyle({'r': 'o', 'R': 'd', 'B': 'x'}[algo])
                        plt.scatter(np.take(x, idx), np.take(y, idx), s=12, label=k, marker=ms,
                                    alpha={'r': 1, 'B': 0.3, 'R': 0.6}[algo],
                                    color=vcol_k[icol])
                        k = None  # display legend only once
                icol += 1
            else:
                i = vct[backend]
                for algo in ['r', 'B', 'R']:
                    idx = np.where(vfill == algo)[0]
                    if len(idx):
                        ms = MarkerStyle(vsymb[i % len(vsymb)])
                        plt.scatter(np.take(x, idx), np.take(y, idx), label=k, marker=ms, color=vcol[backend],
                                    alpha={'r': 1, 'R': 0.6, 'B': 0.3}[algo])
                        k = None  # display legend only once
                vct[backend] += 1

        plt.xlabel("array length")
        plt.ylabel("Theoretical throughput (GBytes/s)")
        plt.ylim(0)
        plt.title(f"{ndim}D FFT (batched)-" + str_config)

        # Use powers of 2 for xticks
        xmin, xmax = plt.xlim()
        step = 2 ** (round(np.log2(xmax - xmin + 1) - 4))
        xmin -= xmin % step
        if xmin < 0:
            xmin = 0
        plt.xticks(np.arange(xmin, xmax, step))

        if one_backend:
            plt.legend(title="Symbols: disc=radix, diamond=Rader, X=Bluestein",
                       title_fontsize=8)
        else:
            plt.legend(title="Alpha: radix=1 Rader=0.6 Bluestein=0.3",
                       title_fontsize=8)
        plt.grid(True)
        plt.tight_layout()
        n = f"pyvkfft-benchmark-{str_config.replace(' ', '_')}-{ndim}D{str_opt}."
        plt.savefig(n + 'svg')
        plt.savefig(n + 'png')
        print(f"Saving {ndim}D benchmark plot to: {n}png and {n}svg")


def run_test(config, args):
    gpu_name = args.gpu
    backend = args.backend
    opencl_platform = None
    verbose = True
    db = args.save
    compare = args.compare
    dbc = None
    dbc0 = None
    first = True

    # Separate parameters for auto-tuning coalescedMemory, aimThreads and warpSize
    vargs = vars(args)
    tune_config = {'backend': {'cuda': 'pycuda', 'opencl': 'pyopencl', 'cupy': 'cupy',
                               'skcuda': 'skcuda', 'gpyfft': 'gpyfft'}[backend]}
    for k in ['coalescedMemory', 'aimThreads', 'warpSize']:
        if len(vargs[k]) > 1:
            tune_config[k] = vargs[k]
        vargs[k] = vargs[k][0]  # We need a scalar
    if len(tune_config) > 1:
        vargs['tune_config'] = tune_config

    for c in config:
        precision = c.precision
        inplace = c.inplace
        r2c = 'r2c' in c.transform
        dct = int(c.transform[-1]) if 'dct' in c.transform else False
        dst = int(c.transform[-1]) if 'dst' in c.transform else False
        sh = tuple(c.shape)
        ndim = c.ndim
        nb_repeat = 4
        nb_loop = c.nb_loop
        gpu_name_real = ''
        platform_name_real = ''
        vkfft_str = None
        nup_str = None
        algo_str = None
        if backend == 'cuda':
            res = bench_pyvkfft_cuda(sh, precision, ndim, nb_repeat, nb_loop, gpu_name, args=vargs,
                                     serial=args.serial, inplace=inplace, r2c=r2c,
                                     dct=dct, dst=dst)
            dt = res['dt']
            gbps = res['gbps']
            gpu_name_real = res['gpu_name_real']
            vkfft_str = res['vkfft_str']
            nup_str = res['nup_str']
            algo_str = res['algo_str']
        elif backend == 'opencl':
            res = bench_pyvkfft_opencl(sh, precision, ndim, nb_repeat, nb_loop, gpu_name,
                                       opencl_platform=opencl_platform, args=vargs,
                                       serial=args.serial, inplace=inplace,
                                       r2c=r2c, dct=dct, dst=dst)
            gbps = res['gbps']
            gpu_name_real = res['gpu_name_real']
            platform_name_real = res['platform_name_real']
            vkfft_str = res['vkfft_str']
            nup_str = res['nup_str']
            algo_str = res['algo_str']
        elif backend == 'skcuda':
            res = bench_skcuda(sh, precision, ndim, nb_repeat, nb_loop, gpu_name, serial=args.serial)
            gbps = res['gbps']
            gpu_name_real = res['gpu_name_real']
        elif backend == 'gpyfft':
            res = bench_gpyfft(sh, precision, ndim, nb_repeat, nb_loop, gpu_name,
                               opencl_platform=opencl_platform,
                               serial=args.serial)
            gbps = res['gbps']
            gpu_name_real = res['gpu_name_real']
            platform_name_real = res['platform_name_real']
        elif backend == 'cupy':
            res = bench_cupy(sh, precision, ndim, nb_repeat, nb_loop, gpu_name, serial=args.serial)
            gbps = res['gbps']
            gpu_name_real = res['gpu_name_real']
        if gpu_name_real is None or gbps == 0:
            # Something went wrong ? Possible timeout ?
            continue
        g = capwords(gpu_name_real.replace('Apple', ''))  # Redundant
        g = g.strip(' _').replace(':', '_')
        plat = capwords(platform_name_real).strip(' _').replace(':', '_')
        if args.bluestein:
            radix = 'BluesteinRader'
        elif args.radix is None:
            radix = 'none'
        else:
            radix = 'x'.join(str(i) for i in args.radix)
        if db:
            if first:
                if type(db) != str:
                    db = f"pyvkfft{__version__}-{vkfft_version()}-" \
                         f"{g.replace(' ', '_')}-{backend}-" \
                         f"{datetime.now().strftime('%Y_%m_%d_%Hh_%Mm_%Ss')}-benchmark.sql"

                hostname = socket.gethostname()
                db = sqlite3.connect(db)
                dbc = db.cursor()

                # For tuned values, use -99 as special value
                coalescedMemory = args.coalescedMemory
                if 'coalescedMemory' in tune_config:
                    if len(tune_config['coalescedMemory']) > 1:
                        coalescedMemory = -99
                aimThreads = args.aimThreads
                if 'aimThreads' in tune_config:
                    if len(tune_config['aimThreads']) > 1:
                        aimThreads = -99
                warpSize = args.warpSize
                if 'warpSize' in tune_config:
                    if len(tune_config['warpSize']) > 1:
                        warpSize = -99

                dbc.execute('CREATE TABLE IF NOT EXISTS config (epoch int, hostname text,'
                            'pyvkfft text, vkfft text, backend text, transform text, radix text,'
                            'precision text, inplace int, gpu text, platform text,'
                            'disableReorderFourStep int, coalescedMemory int, numSharedBanks int,'
                            'aimThreads int, performBandwidthBoost int, registerBoost int,'
                            'registerBoostNonPow2 int, registerBoost4Step int, warpSize int, useLUT int,'
                            'batchedGroup text)')
                dbc.execute('INSERT INTO config VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                            (time.time(), hostname, __version__, vkfft_version(), backend, c.transform,
                             radix, precision, inplace, g, plat, args.disableReorderFourStep,
                             coalescedMemory, args.numSharedBanks,
                             aimThreads, args.performBandwidthBoost, args.registerBoost,
                             args.registerBoostNonPow2, args.registerBoost4Step, warpSize,
                             args.useLUT, 'x'.join(str(i) for i in args.batchedGroup)))
                db.commit()
                dbc.execute('CREATE TABLE IF NOT EXISTS benchmark (epoch int, ndim int, shape text,'
                            'gbps float, algo str, nb_upload str)')
                db.commit()
            dbc.execute('INSERT INTO benchmark VALUES (?,?,?,?,?,?)',
                        (time.time(), ndim, 'x'.join(str(i) for i in sh), gbps,
                         algo_str, nup_str))
            db.commit()
        if compare and first:
            dbc0 = sqlite3.connect(compare).cursor()
        if verbose:
            s = f"{str(c):>40} {gbps:6.1f} GB/s {gpu_name_real} {backend:6^} "
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

            if vkfft_str is not None:
                s += f"[VkFFT: {vkfft_str}]"

            print(s)


def make_parser():
    epilog = "Examples:\n" \
             "* Simple benchmark for radix transforms:\n" \
             "     pyvkfft-benchmark --backend cuda --gpu titan\n\n" \
             "* Systematic benchmark for 1D radix transforms over a given range:\n" \
             "     pyvkfft-benchmark --backend cuda --gpu titan --systematic --ndim 1 --range 2 256\n\n" \
             "* Same but only for powers of 2 and 3 sizes, in 2D, and save the results " \
             "to an SQL file for later plotting:\n" \
             "     pyvkfft-benchmark --backend cuda --gpu titan --systematic --radix 2 3 " \
             "--ndim 2 --range 2 256 --save\n\n" \
             "* plot the result of a benchmark:\n" \
             "     pyvkfft-benchmark --plot pyvkfft-version-gpu-date-etc.sql\n\n" \
             "* plot & compare the results of multiple benchmarks (grouped by 1D/2D/3D transforms):\n" \
             "     pyvkfft-benchmark --plot *.sql\n\n" \
             "* Systematic test in OpenCL for an M1 GPU, tuning the VkFFT algorithm" \
             "  with the best possible 'aimthreads' low-level parameter to maximise" \
             " throughput:\n" \
             "     pyvkfft-benchmark --backend opencl --gpu m1 --systematic --radix --ndim 2 " \
             "--range 2 256 --inplace --aimThreads 16 32 64 --r2c\n\n" \
             "When testing VkFFT, each line also indicates at the end the type of\n" \
             "algorithm used: (r)adix, (R)ader or (B)luestein, the size of the\n" \
             "temporary buffer (if any) and the number of uploads (number of read and\n" \
             "writes) for each axis. \n\nNote 1: the indicated throughput is always\n" \
             "  computed assuming a single read and write for each axis (by convention),\n " \
             "  even if we know the number of uploads is actually larger. \n\n" \
             "Note 2: in the case of DCT1 and DST1 the throughput will be worse as these \n" \
             "  are computed as complex systems of size 2N-2, i.e. with 4x the original size."

    desc = "Run pyvkfft benchmark tests. This is pretty slow as each test runs " \
           "in a separate process (including the GPU initialisation) - this is done to avoid " \
           "any context a memory issues when performing a large number of tests. " \
           "This can also be used to compare results with cufft (scikit-cuda or cupy) and gpyfft. " \
           ""

    parser = argparse.ArgumentParser(prog='pyvkfft-benchmark', epilog=epilog,
                                     description=desc,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--backend', action='store', choices=['cuda', 'opencl', 'gpyfft', 'skcuda', 'cupy'],
                        default='opencl', help="FFT backend to use, 'cuda' and 'opencl' will "
                                               "use pyvkfft with the corresponding language.")
    parser.add_argument('--precision', action='store', choices=['single', 'double'],
                        default='single', help="Precision for the benchmark")
    parser.add_argument('--gpu', action='store', type=str, default=None, help="GPU name (or sub-string)")
    parser.add_argument('--opencl_platform', action='store',
                        help="Name (or sub-string) of the opencl platform to use (case-insensitive). "
                             "Note that by default the PoCL platform is skipped, "
                             "unless it is specifically requested or it is the only one available "
                             "(PoCL has some issues with VkFFT for some transforms)")
    parser.add_argument('--serial', action='store_true',
                        help="Use this to perform all tests in a single process. This is mostly "
                             "useful for testing, and can lead to GPU memory issues, especially "
                             "with skcuda.")
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
    sysgrp.add_argument('--radix', action='store', nargs='*', type=int,
                        help="Perform only radix transforms. Without --radix, all integer "
                             "sizes are tested. With '--radix', all radix transforms allowed "
                             "by the backend are used. Alternatively a list can be given: "
                             "'--radix 2' (only 2**n array sizes), '--radix 2 3 5' "
                             "(only 2**N1 * 3**N2 * 5**N3)",
                        choices=[2, 3, 5, 7, 11, 13], default=None)
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
    sysgrp.add_argument('--minsize-mb', action='store', type=int, default=100,
                        help="Minimal size (in MB) of the transformed array to ensure a precise "
                             "enough timing, as the FT is tested on a stacked array using "
                             "a batch transform. Larger values take more time. Ignored if "
                             "--nbatch is not -1 (the default)")
    sysgrp.add_argument('--nbatch', action='store', type=int,
                        help="Specify the batch size for the array transforms. By default (-1), "
                             "this  number is automatically adjusted for each length so that the total "
                             "size is equal to 'minsize-mb' (100MB by default), e.g. for 2D R2C "
                             "test of 512x512, the batch number is 100. Use 1 to disable "
                             "batch, or any other number to use a fixed batch size.",
                        default=-1)
    sysgrp.add_argument('--r2c', action='store_true',
                        help="Test real-to-complex transform (default is c2c)")
    sysgrp.add_argument('--dct', action='store', default=False, type=int,
                        choices=[1, 2, 3, 4],
                        help="Test direct cosine transform of the given type (default is c2c)")
    sysgrp.add_argument('--dst', action='store', default=False, type=int,
                        choices=[1, 2, 3, 4],
                        help="Test direct sine transform of the given type (default is c2c)")
    sysgrp.add_argument('--inplace', action='store_true',
                        help="Test inplace transforms")

    sysgrp = parser.add_argument_group("advanced", "Advanced options for VkFFT. Do NOT use unless you "
                                                   "really know what these mean. -1 will always "
                                                   "defer the choice to VkFFT. For some parameters "
                                                   "(coalescedMemory, aimThreads and warpSize), if "
                                                   "multiple values are used, this will trigger "
                                                   "the automatic tuning of the transform by testing "
                                                   "each possible configuration of parameters, "
                                                   "before using the optimal parameter for the "
                                                   "actual transform.")
    sysgrp.add_argument('--disableReorderFourStep', action='store', choices=[-1, 0, 1], type=int,
                        default=-1, help="Disables unshuffling of Four step algorithm."
                                         " Requires tempbuffer allocation")
    sysgrp.add_argument('--coalescedMemory', action='store', choices=[-1, 16, 32, 64, 128], type=int,
                        help="Number of bytes to coalesce per one transaction: "
                             "defaults to 32 for Nvidia and AMD, 64 for others."
                             "Should be a power of two", default=[-1], nargs='+')
    sysgrp.add_argument('--numSharedBanks', action='store', choices=[-1] + list(range(16, 64 + 1, 4)), type=int,
                        default=-1, help="Number of shared banks on the target GPU. Default is 32. ")
    sysgrp.add_argument('--aimThreads', action='store', choices=[-1] + list(range(16, 256 + 1, 4)), type=int,
                        default=[-1], help="Try to aim all kernels at this amount of threads. ", nargs='+')
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
                        default=[-1], help="Number of threads per warp/wavefront. Normally automatically "
                                           "derived from the driver. Must be a power of two", nargs='+')
    sysgrp.add_argument('--batchedGroup', action='store', nargs=3, type=int, default=[-1, -1, -1],
                        help="How many FFTs are done per single kernel "
                             "by a dedicated thread block, for each dimension.")
    sysgrp.add_argument('--useLUT', action='store', choices=[-1, 0, 1], type=int,
                        default=-1, help="Use a look-up table to bypass the native sincos functions.")
    sysgrp.add_argument('--forceCallbackVersionRealTransforms', action='store', choices=[-1, 0, 1], type=int,
                        default=-1, help="force callback version of R2C and R2R (DCT/DST) algorithms"
                                         "for all usecases. this is normally activated automatically"
                                         "by VkFFT for odd sizes.")
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.plot:
        plot_benchmark(*args.plot)
        return
    if (args.r2c and (args.dct or args.dst)) or (args.dct and args.dst):
        raise RuntimeError("--r2c, --dct and --dst are mutually exclusive")
    if (args.r2c or args.dct or args.dst) and ('gpyfft' in args.backend or 'skcuda' in args.backend):
        raise RuntimeError("--r2c, --dct and --dst are nor supported for gpyfft or skcuda backends")
    if args.systematic:
        transform = 'r2c' if args.r2c else f'dct{args.dct}' if args.dct else f'dst{args.dst}' if args.dst else 'c2c'
        config = []
        for ndim in args.ndim:
            size_min_max = np.array(args.range_mb) * 1024 ** 2
            if args.precision == 'double':
                size_min_max //= 16
            else:
                size_min_max //= 8
            size_min_max = np.round(size_min_max ** (1 / ndim)).astype(int)

            if args.bluestein:
                if args.radix is not None:
                    raise RuntimeError("--bluestein cannot be used with --radix")
                if args.backend in ['skcuda', 'cupy']:
                    # for cufft, radix transforms only till 7 (and a few undocumented primes up to 127)
                    args.radix = [2, 3, 5, 7]
                else:
                    args.radix = [2, 3, 5, 7, 11, 13]
            elif args.radix is not None:
                if len(args.radix) == 0:  # only --radix was passed
                    if args.backend in ['skcuda', 'cupy']:
                        # for cufft, radix transforms only till 7 (and a few undocumented primes up to 127)
                        args.radix = [2, 3, 5, 7]
                    else:
                        args.radix = [2, 3, 5, 7, 11, 13]

            vshape = np.array(radix_gen_n(nmax=args.range[1], max_size=size_min_max[1],
                                          radix=args.radix, ndim=1, even=False,
                                          nmin=args.range[0], max_pow=None,
                                          range_nd_narrow=None, min_size=size_min_max[0],
                                          inverted=args.bluestein,
                                          r2r=args.dct if args.dct else args.dst),
                              dtype=int).flatten()
            # Number of loops * batch size to transform at least 100 MB
            s = 8
            if args.precision == 'double':
                s *= 2
            elif args.precision == 'half':
                s /= 2
            if args.r2c or args.dct or args.dst:
                s /= 2
            nb = args.minsize_mb * 1024 ** 2 / (vshape ** ndim * 8)

            if args.nbatch == -1:
                nloop, nbatch =  np.ones(len(vshape), dtype=int), nb.astype(int)
            else:
                nbatch = np.ones(len(vshape), dtype=int) * args.nbatch
                nloop = nb.astype(int) / nbatch
            nbatch = np.maximum(1, nbatch).astype(int)
            nloop = np.minimum(1024, np.maximum(1, nloop)).astype(int)
            config += [BenchConfig(transform, [b] + [n] * ndim, ndim, inplace=args.inplace,
                                   precision=args.precision, nb_loop=nl)
                       for b, n, nl in zip(nbatch, vshape, nloop)]
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
