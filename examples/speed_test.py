# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2021- : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

# import pycuda.autoinit
import pycuda.driver as cu_drv
import pycuda.gpuarray as cua
from pyvkfft.cuda import VkFFTApp, _test
import numpy as np
import timeit

cu_drv.init()

# TODO For now this is necessary or the following tests will fail
_test(128)


def speed(shape, ndim, nb=10):
    """
    Perform a speed test using VkFFT (
    :param shape: array shape to use
    :param ndim: number of dimensions for the FFT (e.g. can be 1, 2 or 3 for a 3D array, etc..)
    :param nb: number of repeats for timing
    :return: a tuple with the time per couple of FFT and iFFT, and the idealised memory throughput
        assuming one read and one write of the array per transform axis, in Gbytes/s.
    """
    d = cua.to_gpu(np.random.uniform(0, 1, shape).astype(np.complex64))
    # print(d.shape)
    app = VkFFTApp(d, ndim=ndim)
    app.fft()
    ctx.synchronize()
    ctx.synchronize()
    t0 = timeit.default_timer()
    for i in range(nb):
        app.ifft()
        app.fft()
    ctx.synchronize()
    dt = timeit.default_timer() - t0
    shape = list(shape)
    if len(shape) < 3:
        shape += [1] * (3 - len(shape))
    gbps = d.nbytes * nb * 2 * 2 * 2 / dt / 1024 ** 3
    print("(%4d %4d %4d)[%dD] dt=%6.2f ms %7.2f Gbytes/s" %
          (shape[2], shape[1], shape[0], ndim, dt / nb * 1000, gbps))
    return dt, gbps


ctx = cu_drv.Device(0).make_context()

speed((256, 256, 256), 3)
speed((400, 400, 400), 3)
speed((512, 512, 512), 3)
speed((16, 504, 504), 2)
speed((16, 512, 512), 2)
speed((16, 1024, 1024), 2)
speed((16, 2048, 2048), 2)
speed((1, 512, 512), 2)
speed((1, 1024, 1024), 2)
speed((1, 2600, 2048), 2)
speed((1, 2048, 2600), 2)
speed((8, 2600, 2048), 2)
speed((8, 2048, 2600), 2)
speed((16, 2200, 2400), 2)
speed((16, 2310, 2730), 2)
speed((16, 2730, 2310), 2)  # 2310=2*3*5*7*11, 2730=2*3*5*7*13
speed((16, 2400, 2200), 2)
speed((1, 3080, 3080), 2)
speed((8, 3072, 3072), 2)

ctx.pop()
