# -*- coding: utf-8 -*-

# PyVkFFT
#   (c) 2023- : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

from copy import deepcopy
import itertools
import timeit
import numpy as np


def tune_vkfft(tune, shape, dtype: type, ndim=None, inplace=True, stream=None, queue=None,
               norm=1, r2c=False, dct=False, dst=False, axes=None, strides=None,
               verbose=False, r2c_odd=False, **kwargs):
    """
    Automatically test different configurations for a VkFFTApp, returning
    the set of parameters which maximise the FT throughput.
    The three parameters which are recommended to optimise are aimThreads,
    warpSize and coalescedMemory. Usually tuning a single one should suffice,
    but the right one could depend on the backend and GPU brand.

    Note that the GPU context must have been initialised before calling
    this function.

    :param tune: dictionary including the backend used and the parameter
        values which will be tested.
        This is EXPERIMENTAL, as wrong parameters may lead to crashes.
        Note that this will allocate temporary GPU arrays, unless the arrays
        to used have been passed as parameters ('dest' and 'src').
        Examples:
         * tune={'backend':'cupy'} - minimal example, will automatically test a small
           set of parameters (4 to 10 tests). Recommended !
         * tune={'backend':'pycuda', 'warpSize':[8,16,32,64,128]}: this will test
           5 possible values for the warpSize.
         * tune={'backend':'pyopencl', 'aimThreads':[32,64,128,256]}: this will test
           5 possible values for the warpSize.
         * tune={'backend':'cupy', 'groupedBatch':[[-1,-1,-1],[8,8,8], [4,16,16}:
           this will test 3 possible values for groupedBatch. This one is more
           tricky to use.
         * tune={'backend':'cupy, 'warpSize':[8,16,32,64,128], 'src':a}: this
           will test 5 possible values for the warpSize, with a given source GPU
           array. This would only be valid for an inplace transform as no
           destination array is given.
    :param shape: the shape of the array to be transformed. The number
        of dimensions of the array can be larger than the FFT dimensions,
        but only for 1D and 2D transforms. 3D FFT transforms can only
        be done on 3D arrays.
    :param dtype: the numpy dtype of the source array (can be complex64 or complex128)
    :param ndim: the number of dimensions to use for the FFT. By default,
        uses the array dimensions. Can be smaller, e.g. ndim=2 for a 3D
        array to perform a batched 3D FFT on all the layers. The FFT
        is always performed along the last axes if the array's number
        of dimension is larger than ndim, i.e. on the x-axis for ndim=1,
        on the x and y axes for ndim=2.
    :param inplace: if True (the default), performs an inplace transform and
        the destination array should not be given in fft() and ifft().
    :param stream: the pycuda.driver.Stream or cupy.cuda.Stream to use
        for the transform. This can also be the pointer/handle (int) to the
        cuda stream object. If None, the default stream will be used.
    :param queue: the pyopencl CommandQueue to use for the transform.
    :param norm: if 0 (unnormalised), every transform multiplies the L2
        norm of the array by its size (or the size of the transformed
        array if ndim<d.ndim).
        if 1 (the default) or "backward", the inverse transform divides
        the L2 norm by the array size, so FFT+iFFT will keep the array norm.
        if "ortho", each transform will keep the L2 norm, but that will
        involve an extra read & write operation.
    :param r2c: if True, will perform a real->complex transform, where the
        complex destination is a half-hermitian array.
        For an inplace transform, if the input data shape is (...,nx), the input
        float array should have a shape of (..., nx+2), the last two columns
        being ignored in the input data, and the resulting
        complex array (using pycuda's GPUArray.view(dtype=np.complex64) to
        reinterpret the type) will have a shape (..., nx//2 + 1).
        For an out-of-place transform, if the input (real) shape is (..., nx),
        the output (complex) shape should be (..., nx//2+1).
        Note that for C2R transforms with ndim>=2, the source (complex) array
        is modified.
    :param dct: used to perform a Direct Cosine Transform (DCT) aka a R2R transform.
        An integer can be given to specify the type of DCT (1, 2, 3 or 4).
        if dct=True, the DCT type 2 will be performed, following scipy's convention.
    :param dst: used to perform a Direct Cosine Transform (DST) aka a R2R transform.
        An integer can be given to specify the type of DST (1, 2, 3 or 4).
        if dst=True, the DST type 2 will be performed, following scipy's convention.
    :param axes: a list or tuple of axes along which the transform should be made.
        if None, the transform is done along the ndim fastest axes, or all
        axes if ndim is None. For R2C transforms, the fast axis must be
        transformed.
    :param strides: the array strides - needed if not C-ordered.
    :param verbose: if True, print speed for each configuration
    :param r2c_odd: this should be set to True to perform an inplace r2c/c2r
        transform with an odd-sized fast (x) axis.
        Explanation: to perform a 1D inplace transform of an array with 100
            elements, the input array should have a 100+2 size, resulting in
            a half-Hermitian array of size 51. If the input data has a size
            of 101, the input array should also be padded to 102 (101+1), and
            the resulting half-Hermitian array also has a size of 51. A
            flag is thus needed to differentiate the cases of 100+2 or 101+1.
    :param: extra parameters passed on to VkFFT

    :raises RuntimeError: if the optimisation fails
    :return: (kw, res) where kw are the optimal kwargs which can be passed to the
        VkFFTApp creation routine, and res is the full set of results for the
        different configurations tested.
    """
    try:
        # Import
        if tune['backend'] == 'cupy':
            import cupy as cua
            from cupy.cuda import Event
            from .cuda import VkFFTApp
        elif tune['backend'] == 'pycuda':
            import pycuda.gpuarray as cua
            from pycuda.driver import Event
            from .cuda import VkFFTApp
        else:
            from .opencl import VkFFTApp
            import pyopencl as cl
            import pyopencl.array as cla
        # GPU arrays
        if 'src' not in tune:
            if tune['backend'] == 'pyopencl':
                src = cla.to_device(queue, np.ones(shape, dtype=dtype))
            else:
                src = cua.ones(shape, dtype=dtype)
        else:
            src = tune['src']
        if inplace:
            dest = src
        else:
            if 'dest' not in tune:
                if 'src' in tune:
                    raise RuntimeError("VkFFT autotune: 'src' array is provided but not the destination, "
                                       "which is required for an out-of-place transform")
                if r2c:
                    # TODO: handle different strides, with fast axis != -1
                    shc = list(shape)
                    shc[-1] = shape[-1] // 2 + 1
                    shc = tuple(shc)
                    dtypec = np.complex64 if dtype == np.float32 else np.complex128
                    if tune['backend'] == 'pyopencl':
                        dest = cla.to_device(queue, np.ones(shc, dtype=dtypec))
                    else:
                        dest = cua.ones(shc, dtype=np.float32).astype(dtypec)
                else:
                    dest = src.copy()
            else:
                dest = tune['dest']

        # Parameters to test
        vk = []
        for k in tune.keys():
            if k in ['disableReorderFourStep', 'coalescedMemory', 'numSharedBanks',
                     'aimThreads', 'performBandwidthBoost', 'registerBoost',
                     'registerBoostNonPow2', 'registerBoost4Step', 'warpSize',
                     'groupedBatch']:
                vk.append(k)

        if len(vk) == 0:
            # Only the backend was supplied, choose automatically the parameters to tune
            tune = deepcopy(tune)
            if tune['backend'] in ['cupy', 'pycuda']:
                vk.append('coalescedMemory')
                tune['coalescedMemory'] = [32, 64, 128]
            else:
                # pyopencl - choose tuning parameters based on platform
                if 'apple' in queue.device.name.lower():
                    vk.append('aimThreads')
                    tune['aimThreads'] = [32, 64, 128, 256]
                elif 'nvidia' in queue.device.platform.name.lower():
                    vk.append('coalescedMemory')
                    tune['coalescedMemory'] = [32, 64, 128]
                else:
                    # TODO: try other GPU/platforms
                    vk.append('aimThreads')
                    tune['aimThreads'] = [32, 64, 128, 256]

        if verbose:
            print('VkFFT parameters to tune:  ' + '  '.join(vk))
        res = []
        args = (tune[k] for k in vk)
        for v in itertools.product(*args):
            kw = deepcopy(kwargs)
            for i in range(len(v)):
                kw[vk[i]] = v[i]
            if tune['backend'] == 'pyopencl':
                app = VkFFTApp(shape, dtype=dtype, queue=queue, ndim=ndim, inplace=inplace,
                               norm=norm, r2c=r2c, dct=dct, dst=dst, axes=axes,
                               strides=strides, r2c_odd=r2c_odd, **kw)
            else:
                app = VkFFTApp(shape, dtype=dtype, ndim=ndim, inplace=inplace, stream=stream,
                               norm=norm, r2c=r2c, dct=dct, dst=dst, axes=axes,
                               strides=strides, r2c_odd=r2c_odd, **kw)
                start = Event()
                stop = Event()
            dt = 0
            # Repeat the transform enough to have a meaningful measurement
            niter = max(1, int(1e8 // src.nbytes))
            for i in range(3):  # Get best of 3 transform speed
                if tune['backend'] == 'pyopencl':
                    queue.finish()
                    t0 = timeit.default_timer()
                    for ii in range(niter):
                        dest = app.fft(src, dest)
                        src = app.ifft(dest, src)
                    queue.finish()
                    dt1 = timeit.default_timer() - t0
                else:
                    start.record()
                    for ii in range(niter):
                        dest = app.fft(src, dest)
                        src = app.ifft(dest, src)
                    stop.record()
                    stop.synchronize()
                    if tune['backend'] == 'cupy':
                        dt1 = cua.cuda.get_elapsed_time(start, stop) / 1000
                    else:
                        dt1 = stop.time_since(start) / 1000
                if dt == 0:
                    dt = dt1
                elif dt1 < dt:
                    dt = dt1

            gbps = niter * src.nbytes * app.ndim * 2 * 2 / dt / 1024 ** 3
            del app
            res.append((kw, gbps, dt))
            if verbose:
                s = f"VkFFT tune {shape}"
                for ii in range(len(v)):
                    s += f" {vk[ii]}={v[ii]}"
                s += f" dt={dt:.3f} {gbps:.3f} GB/s"
                print(s)
        res.sort(key=lambda x: x[1])
        return res[-1][0], res
    except Exception as ex:
        print(ex)
        import traceback
        print(traceback.format_exc())
        return kwargs, []
