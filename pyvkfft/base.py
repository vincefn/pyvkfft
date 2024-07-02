# -*- coding: utf-8 -*-

# PyVkFFT
#   (c) 2021- : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import os
import platform
import sysconfig
import ctypes
import warnings
from enum import Enum
import numpy as np
from . import config

# np.complex32 does not exist yet https://github.com/numpy/numpy/issues/14753
complex32 = np.dtype([('re', np.float16), ('im', np.float16)])

# Type to pass array size, omit and batch as int arrays
ctype_int_size_p = np.ctypeslib.ndpointer(dtype=int, ndim=1, flags='C_CONTIGUOUS')


class VkFFTResult(Enum):
    """ VkFFT error codes from vkFFT.h """
    VKFFT_SUCCESS = 0
    VKFFT_ERROR_MALLOC_FAILED = 1
    VKFFT_ERROR_INSUFFICIENT_CODE_BUFFER = 2
    VKFFT_ERROR_INSUFFICIENT_TEMP_BUFFER = 3
    VKFFT_ERROR_PLAN_NOT_INITIALIZED = 4
    VKFFT_ERROR_NULL_TEMP_PASSED = 5
    VKFFT_ERROR_INVALID_PHYSICAL_DEVICE = 1001
    VKFFT_ERROR_INVALID_DEVICE = 1002
    VKFFT_ERROR_INVALID_QUEUE = 1003
    VKFFT_ERROR_INVALID_COMMAND_POOL = 1004
    VKFFT_ERROR_INVALID_FENCE = 1005
    VKFFT_ERROR_ONLY_FORWARD_FFT_INITIALIZED = 1006
    VKFFT_ERROR_ONLY_INVERSE_FFT_INITIALIZED = 1007
    VKFFT_ERROR_INVALID_CONTEXT = 1008
    VKFFT_ERROR_INVALID_PLATFORM = 1009
    VKFFT_ERROR_ENABLED_saveApplicationToString = 1010,
    VKFFT_ERROR_EMPTY_FFTdim = 2001
    VKFFT_ERROR_EMPTY_size = 2002
    VKFFT_ERROR_EMPTY_bufferSize = 2003
    VKFFT_ERROR_EMPTY_buffer = 2004
    VKFFT_ERROR_EMPTY_tempBufferSize = 2005
    VKFFT_ERROR_EMPTY_tempBuffer = 2006
    VKFFT_ERROR_EMPTY_inputBufferSize = 2007
    VKFFT_ERROR_EMPTY_inputBuffer = 2008
    VKFFT_ERROR_EMPTY_outputBufferSize = 2009
    VKFFT_ERROR_EMPTY_outputBuffer = 2010
    VKFFT_ERROR_EMPTY_kernelSize = 2011
    VKFFT_ERROR_EMPTY_kernel = 2012
    VKFFT_ERROR_EMPTY_applicationString = 2013,
    VKFFT_ERROR_UNSUPPORTED_RADIX = 3001
    VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH = 3002
    VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH_R2C = 3003
    VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH_R2R = 3004
    VKFFT_ERROR_UNSUPPORTED_FFT_OMIT = 3005
    VKFFT_ERROR_FAILED_TO_ALLOCATE = 4001
    VKFFT_ERROR_FAILED_TO_MAP_MEMORY = 4002
    VKFFT_ERROR_FAILED_TO_ALLOCATE_COMMAND_BUFFERS = 4003
    VKFFT_ERROR_FAILED_TO_BEGIN_COMMAND_BUFFER = 4004
    VKFFT_ERROR_FAILED_TO_END_COMMAND_BUFFER = 4005
    VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE = 4006
    VKFFT_ERROR_FAILED_TO_WAIT_FOR_FENCES = 4007
    VKFFT_ERROR_FAILED_TO_RESET_FENCES = 4008
    VKFFT_ERROR_FAILED_TO_CREATE_DESCRIPTOR_POOL = 4009
    VKFFT_ERROR_FAILED_TO_CREATE_DESCRIPTOR_SET_LAYOUT = 4010
    VKFFT_ERROR_FAILED_TO_ALLOCATE_DESCRIPTOR_SETS = 4011
    VKFFT_ERROR_FAILED_TO_CREATE_PIPELINE_LAYOUT = 4012
    VKFFT_ERROR_FAILED_SHADER_PREPROCESS = 4013
    VKFFT_ERROR_FAILED_SHADER_PARSE = 4014
    VKFFT_ERROR_FAILED_SHADER_LINK = 4015
    VKFFT_ERROR_FAILED_SPIRV_GENERATE = 4016
    VKFFT_ERROR_FAILED_TO_CREATE_SHADER_MODULE = 4017
    VKFFT_ERROR_FAILED_TO_CREATE_INSTANCE = 4018
    VKFFT_ERROR_FAILED_TO_SETUP_DEBUG_MESSENGER = 4019
    VKFFT_ERROR_FAILED_TO_FIND_PHYSICAL_DEVICE = 4020
    VKFFT_ERROR_FAILED_TO_CREATE_DEVICE = 4021
    VKFFT_ERROR_FAILED_TO_CREATE_FENCE = 4022
    VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_POOL = 4023
    VKFFT_ERROR_FAILED_TO_CREATE_BUFFER = 4024
    VKFFT_ERROR_FAILED_TO_ALLOCATE_MEMORY = 4025
    VKFFT_ERROR_FAILED_TO_BIND_BUFFER_MEMORY = 4026
    VKFFT_ERROR_FAILED_TO_FIND_MEMORY = 4027
    VKFFT_ERROR_FAILED_TO_SYNCHRONIZE = 4028
    VKFFT_ERROR_FAILED_TO_COPY = 4029
    VKFFT_ERROR_FAILED_TO_CREATE_PROGRAM = 4030
    VKFFT_ERROR_FAILED_TO_COMPILE_PROGRAM = 4031
    VKFFT_ERROR_FAILED_TO_GET_CODE_SIZE = 4032
    VKFFT_ERROR_FAILED_TO_GET_CODE = 4033
    VKFFT_ERROR_FAILED_TO_DESTROY_PROGRAM = 4034
    VKFFT_ERROR_FAILED_TO_LOAD_MODULE = 4035
    VKFFT_ERROR_FAILED_TO_GET_FUNCTION = 4036
    VKFFT_ERROR_FAILED_TO_SET_DYNAMIC_SHARED_MEMORY = 4037
    VKFFT_ERROR_FAILED_TO_MODULE_GET_GLOBAL = 4038
    VKFFT_ERROR_FAILED_TO_LAUNCH_KERNEL = 4039
    VKFFT_ERROR_FAILED_TO_EVENT_RECORD = 4040
    VKFFT_ERROR_FAILED_TO_ADD_NAME_EXPRESSION = 4041
    VKFFT_ERROR_FAILED_TO_INITIALIZE = 4042
    VKFFT_ERROR_FAILED_TO_SET_DEVICE_ID = 4043
    VKFFT_ERROR_FAILED_TO_GET_DEVICE = 4044
    VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT = 4045
    VKFFT_ERROR_FAILED_TO_CREATE_PIPELINE = 4046
    VKFFT_ERROR_FAILED_TO_SET_KERNEL_ARG = 4047
    VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE = 4048
    VKFFT_ERROR_FAILED_TO_RELEASE_COMMAND_QUEUE = 4049
    VKFFT_ERROR_FAILED_TO_ENUMERATE_DEVICES = 4050
    VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE = 4051
    VKFFT_ERROR_FAILED_TO_CREATE_EVENT = 4052


def _library_path(basename):
    if platform.system() == 'Windows':
        # We patched build_ext so the module is a .so and not a dll
        ext = '.so'
    else:
        ext = sysconfig.get_config_var('EXT_SUFFIX')
    return os.path.join(os.path.dirname(__file__) or os.path.curdir, basename + ext)


def load_library(basename):
    return ctypes.cdll.LoadLibrary(_library_path(basename))


def primes(n):
    """ Returns the prime decomposition of n as a list.
    This only remains as a useful function, but VkFFT
    allows any prime decomposition, even if performance
    can be better for prime numbers <=13.
    """
    v = [1]
    assert (n > 0)
    i = 2
    while i * i <= n:
        while n % i == 0:
            v.append(i)
            n //= i
        i += 1
    if n > 1:
        v.append(n)
    return v


def radix_gen(nmax, radix, even=False, exclude_one=True, inverted=False,
              nmin=None, max_pow=None, r2r=False):
    """
    Generate an array of integers which are only multiple of powers
    of base integers, e.g. 2**N1 * 3**N2 * 5**N3 etc...

    :param nmax: the maximum integer to return (included)
    :param radix: the list/tuple of base integers - which don't need
        to be primes
    :param even: if True, only return even numbers
    :param exclude_one: if True (the default), exclude 1
    :param inverted: if True, the returned array will only include
        integers which are NOT in the form 2**N1 * 3**N2 * 5**N3...
    :param nmin: if not None, the integer values returned will be >=nmin
    :param max_pow: if not None, the N1, N2, N3... powers (for sizes
        in the form 2**N1 * 3**N2 * 5**N3) will be at max equal to this
        value, which allows to reduce the number of generated sizes
        while testing all base radixes
    :param r2r: if r2r=4, assume we try to generate radix sizes for
        a DCT1 or DST1 transform, which are computed as a C2C
        transform of size 2N-2.
        If r2r=4, assume we try to generate radix sizes for a
        DCT4 or DST4, which are performed as a C2C transform
        of size n/2 (if n is even) or n (if odd).
    :return: the numpy array of integers, sorted
    """
    a = np.ones(1, dtype=np.int64)
    for i in range(len(radix)):
        if max_pow is None:
            tmp = np.arange(int(np.floor(np.log(nmax) / np.log(radix[i]))) + 1)
        else:
            tmp = np.arange(min(max_pow, int(np.floor(np.log(nmax) / np.log(radix[i])))) + 1)
        a = a * radix[i] ** tmp[:, np.newaxis]
        a = a.flatten()
        a = a[a <= nmax]
    if r2r == 1:
        # We need 2N-2 to be radix, so use generated n+1
        a = a[a > 1] + 1
    elif r2r == 4:
        a[a % 2 == 0] //= 2
        a = np.unique(a)
    a = a[a <= nmax]

    if inverted:
        b = np.arange(nmax + 1)
        b[a] = 0
        a = b.take(np.nonzero(b)[0])
    if even:
        a = a[(a % 2) == 0]
    if nmin is not None:
        a = a[a >= nmin]
    a.sort()
    a = a[a > 0]  # weird cases with r2r=1 or 4
    if len(a):
        if exclude_one and a[0] == 1:
            return a[1:]
    return a


def radix_gen_n(nmax, max_size, radix, ndim=None, even=False, exclude_one=True, inverted=False,
                nmin=None, max_pow=None, range_nd_narrow=None, min_size=0, r2r=False):
    """
    Generate a list of array shape with integers which are only multiple
    of powers of base integers, e.g. 2**N1 * 3**N2 * 5**N3 etc...,
    for each of the dimensions, and with a maximum size.
    Note that this can generate a large number of sizes.

    :param nmax: the maximum value for the length of each dimension (included)
    :param max_size: the maximum size (number of elements) for the array.
    :param radix: the list/tuple of base integers - which don't need
        to be primes. If None, all sizes are allowed
    :param ndim: the number of dimensions allowed. If None, 1D, 2D and 3D
        shapes are mixed.
    :param even: if True, only return even numbers
    :param exclude_one: if True (the default), exclude 1
    :param inverted: if True, the returned array will only include
        integers which are NOT in the form 2**N1 * 3**N2 * 5**N3...
    :param nmin: if not None, the integer values returned will be >=nmin
    :param max_pow: if not None, the N1, N2, N3... powers (for sizes
        in the form 2**N1 * 3**N2 * 5**N3) will be at max equal to this
        value, which allows to reduce the number of generated sizes
        while testing all base radixes
    :param range_nd_narrow: if a tuple of values (drel, dabs) is given,
        with drel within [0;1], for dimensions>1,
        in an array of shape (s0, s1, s2), the difference of lengths
        with respect to the first dimension cannot be larger than
        min(drel * s0, dabs). This allows to reduce the number
        of shapes tested. With drel=dabs=0, all dimensions must
        have identical lengths.
    :param min_size: the minimum size (number of elements). This can be
        used to separate large array tests and use a larger number of
        parallel process for smaller ones.
    :param r2r: if r2r=4, assume we try to generate radix sizes for
        a DCT1 or DST1 transform, which are computed as a C2C
        transform of size 2N-2.
        If r2r=4, assume we try to generate radix sizes for a
        DCT4 or DST4, which are performed as a C2C transform
        of size n/2 (if n is even) or n (if odd).
    :return: the list of array shapes.
    """
    v = []
    if radix is None:
        if even:
            if nmin is None:
                n0 = np.arange(2, min(nmax, max_size), 2)
            else:
                n0 = np.arange(nmin + nmin % 2, min(nmax, max_size), 2)
        else:
            if nmin is None:
                n0 = np.arange(2, min(nmax, max_size))
            else:
                n0 = np.arange(nmin, min(nmax, max_size))
    else:
        n0 = radix_gen(nmax, radix, even=even, exclude_one=exclude_one, inverted=inverted,
                       nmin=nmin, max_pow=max_pow, r2r=r2r)

    if ndim is None or ndim in [1, 12, 123]:
        idx = np.nonzero((n0 <= max_size) * (n0 >= min_size))[0]
        if len(idx):
            v += list(zip(n0.take(idx)))
    if ndim is None or ndim in [2, 12, 123]:
        vidx = list(range(0, len(n0), 1000))
        if vidx[-1] != len(n0):
            vidx.append(len(n0))
        for i1 in range(len(vidx) - 1):
            l01 = n0[vidx[i1]:vidx[i1 + 1]]
            for i2 in range(len(vidx) - 1):
                l2 = n0[vidx[i2]:vidx[i2 + 1]][:, np.newaxis]
                s = (l01 * l2).flatten()
                l1, l2 = (l01 + np.zeros_like(l2)).flatten(), (l2 + np.zeros_like(l01)).flatten()
                tmp = (s <= max_size) * (s >= min_size)
                if range_nd_narrow is not None:
                    drel, dabs = range_nd_narrow
                    m = np.maximum(dabs, l1 * drel)
                    tmp = np.logical_and(tmp, abs(l1 - l2) <= m)
                idx = np.nonzero(tmp)[0]
                if len(idx):
                    v += list(zip(l1.take(idx), l2.take(idx)))
    if ndim is None or ndim in [3, 123]:
        vidx = list(range(0, len(n0), 100))
        if vidx[-1] != len(n0):
            vidx.append(len(n0))
        for i1 in range(len(vidx) - 1):
            l01 = n0[vidx[i1]:vidx[i1 + 1]]
            for i2 in range(len(vidx) - 1):
                l02 = n0[vidx[i2]:vidx[i2 + 1], np.newaxis]
                for i3 in range(len(vidx) - 1):
                    l3 = n0[vidx[i3]:vidx[i3 + 1], np.newaxis, np.newaxis]
                    # print(i1, i2, i3, l1.shape, l2.shape, l3.shape)
                    s = (l01 * l02 * l3).flatten()
                    l1, l2, l3 = (l01 + np.zeros_like(l02) + np.zeros_like(l3)).flatten(), \
                        (l02 + np.zeros_like(l01) + np.zeros_like(l3)).flatten(), \
                        (l3 + np.zeros_like(l01) + np.zeros_like(l02)).flatten()
                    tmp = (s <= max_size) * (s >= min_size)
                    if range_nd_narrow is not None:
                        drel, dabs = range_nd_narrow
                        m = np.maximum(dabs, l1 * drel)
                        tmp = np.logical_and(tmp, (abs(l1 - l2) <= m)
                                             * (abs(l1 - l3) <= m))
                    idx = np.nonzero(tmp)[0]
                    if len(idx):
                        v += list(zip(l1.take(idx), l2.take(idx), l3.take(idx)))
    return v


def strides_nonzero(strides):
    """
    Fix the strides for an array, if one is zero it should be set to the
    smallest stride between the next and previous nonzero stride.
    """
    if strides is None or np.isscalar(strides):
        return strides
    n = len(strides)
    strides = list(strides)
    for i in range(n):
        s = strides[i]
        if s == 0:
            # use the maximum previous or next nonzero stride
            m1, m2 = 0, 0
            for ii in range(i + 1, n - 1):
                if strides[ii] > 0:
                    m1 = strides[ii]
                    break
            for ii in range(i - 1, -1, -1):
                if strides[ii] > 0:
                    m2 = strides[ii]
                    break
            strides[i] = min(m1, m2)
    return strides


def calc_transform_axes(shape, axes=None, ndim=None, strides=None):
    """ Compute the final shape of the array to be passed
    to VkFFT, and the axes for which the transform should
    be skipped.
    By collapsing non-transformed consecutive axes and using batch transforms,
    it is possible to support larger dimensions (the limit is set at
    compilation time).

    :param shape: the initial shape of the data array. Note that this shape
        should be in the usual numpy order, i.e. the fastest axis is
        listed last. e.g. (nz, ny, nx)
    :param axes: the axes to be transformed. if None, all axes
        are transformed, or up to ndim.
    :param ndim: the number of dimensions for the transform. If None,
        the number of axes is used
    :param strides: the array strides. If None, a C-order is assumed
        with the fastest axes along the last dimensions (numpy default)
    :return: (shape, n_batch, skip_axis, ndim, axes0) with the shape after collapsing
        consecutive non-transformed axes (padded with ones if necessary,
        with the order adequate for VkFFT i.e. (nx, ny, nz,...),
        the batch size (e.g. 5 if shape=(5,16,16) and ndim=2),
        the list of booleans indicating which axes should
        be skipped, and the number of transform axes. Finally, axes0
        is returned as a list of transformed axes, before any axis collapsing.
    """
    # reverse order to have list as (nx, ny, nz,...)
    shape1 = list(reversed(list(shape)))
    n = len(shape)
    if np.isscalar(axes):
        axes = [axes]
    if ndim is None:
        if axes is None:
            ndim1 = len(shape)
            axes = list(range(ndim1))
    else:
        ndim1 = ndim
        if axes is not None:
            if ndim1 != len(axes):
                raise RuntimeError("The number of transform axes does not match ndim:", axes, ndim)

    # Axes with VkFFT order
    if axes is not None:
        axes1 = [n - 1 - ax if ax >= 0 else -ax - 1 for ax in axes]
    else:
        # will be modified to take fastest axes if F-ordered
        axes1 = list(range(ndim1))

    if strides is not None and n > 1:
        # The fast axis must come first for VkFFT.
        # Careful with the special case of axes with size 1 (stride does not increase or is 0...)
        # Try with shape=(3,1), F and C-ordered as a corner case...
        s0 = strides_nonzero(strides)
        reorder_shape, reorder_axes = False, False
        if not np.all([s0[i] >= s0[i + 1] for i in range(n - 1)]):
            # Array is F-ordered, need to reorder shape & axes
            if axes is None:
                # Using ndim, so axes1 is already OK
                reorder_shape = True
            else:
                reorder_shape, reorder_axes = True, True
        elif axes is not None and (np.sum([s > 1 for s in shape]) == 1):
            # Case from hell: e.g. shape=(3,1). Strides won't help, so use axes
            if axes1[-1] == n - 1:  # same as axes[-1]==0
                reorder_shape, reorder_axes = True, True
        if reorder_shape:
            shape1 = shape1[::-1]
        if reorder_axes:
            axes1 = [n - 1 - ax if ax >= 0 else -ax - 1 for ax in axes1]

    # List of non-transformed axes
    skip_axis = [True] * len(shape1)
    for i in axes1:
        skip_axis[i] = False

    if np.all([shape1[ax] == 1 for ax in axes1]):
        raise RuntimeError(f"No axis is actually transformed: shape={shape} ndim={ndim} "
                           f"axes={axes} strides={strides} vkfft_shape={shape1} "
                           f"vkfft_axes={axes1} vkfft_skip={skip_axis}")

    # Collapse non-transform axes when possible
    i = 0
    while i <= len(shape1) - 2:
        if skip_axis[i] and skip_axis[i + 1]:
            shape1[i] *= shape1[i + 1]
            shape1.pop(i + 1)
            skip_axis.pop(i + 1)
        else:
            i += 1

    # Fix ndim so skipped axes are counted
    ndim1 = len(shape1) - list(reversed(skip_axis)).index(False)

    # For VkFFT > cc2b427, all dimensions beyond the
    # transformed axes should be in n_batch
    n_batch = 1
    for i in range(ndim1, len(shape1)):
        n_batch *= shape1[i]
        shape1[i] = 1
        # Axes beyond ndim are marked skipped
        skip_axis[i] = True

    if axes is None:
        # Return the actual axes transformed
        axes = [n - 1 - ax if ax >= 0 else -ax - 1 for ax in axes1]

    return shape1, n_batch, skip_axis, ndim1, axes


def check_vkfft_result(res, shape=None, dtype=None, ndim=None, inplace=None,
                       norm=None, r2c=None, dct=None, dst=None, axes=None, backend=None,
                       strides=None, vkfft_shape=None, vkfft_skip=None, vkfft_nbatch=None):
    """
    Check VkFFTResult code.

    :param res: the result code from launching a transform.
    :param shape: shape of the array
    :param dtype: data type of the array
    :param ndim: number of transform dimensions
    :param inplace: True or False
    :param norm: 0 or1 or "ortho"
    :param r2c: True or False
    :param dct: False, 1, 2, 3 or 4
    :param dst: False, 1, 2, 3 or 4
    :param axes: transform axes
    :param backend: the backend
    :param strides: the array strides
    :param vkfft_shape: the shape passed to VkFFT
    :param vkfft_skip: the skipped axis list passed to VkFFT
    :param vkfft_nbatch: VkFFT batch parameter
    :raises RuntimeError: if res != 0
    """
    if isinstance(res, ctypes.c_int):
        res = res.value
    if res != 0:
        s = ""
        if r2c:
            s += "R2C "
        elif dct:
            s += "DCT%d " % dct
        elif dst:
            s += "DST%d " % dst
        else:
            s += "C2C "
        if r2c and inplace and shape is not None:
            tmp = list(shape)
            tmp[-1] -= 2
            shstr = str(tuple(tmp)).replace(" ", "")
            if ",)" in shstr:
                s += shstr.replace(",)", "+2)") + " "
            else:
                s += shstr.replace(")", "+2)") + " "
        else:
            s += str(shape).replace(" ", "") + " "
        if dtype is not None:
            s += str(dtype) + " "
        if axes is not None:
            s += str(axes).replace(" ", "") + " "
        if strides is not None:
            s += "strides=" + str(strides).replace(" ", "") + " "
        if ndim is not None:
            s += "%dD " % ndim
        if inplace:
            s += "inplace "
        if norm:
            s += "norm=%s " % str(norm)
        if vkfft_shape is not None or vkfft_skip is not None or vkfft_nbatch is not None:
            s += "[VkFFT:"
            if vkfft_shape is not None:
                s += " shape= " + str(vkfft_shape).replace(" ", "")
            if vkfft_skip is not None:
                s += " skip=" + str([int(sk) for sk in vkfft_skip]).replace(" ", "")
            if vkfft_nbatch is not None:
                s += f" nbatch={vkfft_nbatch}"
            s += "] "
        if backend is not None:
            s += "[%s]" % backend
        try:
            r = VkFFTResult(res)
            raise RuntimeError("VkFFT error %d: %s %s" % (res, r.name, s))
        except ValueError:
            raise RuntimeError("VkFFT error %d (unknown) %s" % (res, s))


class VkFFTApp:
    """
    VkFFT application interface implementing a FFT plan, base implementation
    handling functions and paremeters common to the CUDA and OpenCL backends.
    """

    def __init__(self, shape, dtype: type, ndim=None, inplace=True, norm=1,
                 r2c=False, dct=False, dst=False, axes=None, strides=None,
                 r2c_odd=False, **kwargs):
        """
        Init function for the VkFFT application.

        :param shape: the shape of the array to be transformed. The number
            of dimensions of the array can be larger than the FFT dimensions.
        :param dtype: the numpy dtype of the source array (can be complex64 or complex128)
        :param ndim: the number of dimensions to use for the FFT. By default,
            uses the array dimensions. Can be smaller, e.g. ndim=2 for a 3D
            array to perform a batched 3D FFT on all the layers. The FFT
            is always performed along the last axes if the array's number
            of dimension is larger than ndim, i.e. on the x-axis for ndim=1,
            on the x and y axes for ndim=2, etc.. Unless axes are given.
        :param inplace: if True (the default), performs an inplace transform and
            the destination array should not be given in fft() and ifft().
        :param norm: if 0 (unnormalised), every transform multiplies the L2
            norm of the array by its size (or the size of the transformed
            array if ndim<d.ndim).
            if 1 (the default) or "backward", the inverse transform divides
            the L2 norm by the array size, so FFT+iFFT will keep the array norm.
            if "ortho", each transform will keep the L2 norm, but that will
            involve an extra read & write operation.
        :param r2c: if True, will perform a real->complex transform, where the
            complex destination is a half-hermitian array.
            For an inplace transform, if the transformed data shape is (...,nx),
            the input float array should have a shape of (..., nx+2), the last
            two columns being ignored in the input data, and the resulting
            complex array (using pycuda's GPUArray.view(dtype=np.complex64) to
            reinterpret the type) will have a shape (..., nx//2 + 1).
            For an out-of-place transform, if the input (real) shape is (..., nx),
            the output (complex) shape should be (..., nx//2+1).
            Note 1: the above shape changes are true for C-contiguous arrays;
            generally the axis which is halved by the R2C transform always is
            the fast axis -with a stride of 1 element. For F-contiguous arrays
            this will be the first dimension instead of the last.
            Note 2:for C2R transforms with ndim>=2, the source (complex) array
            is modified.
        :param dct: used to perform a Direct Cosine Transform (DCT) aka a R2R transform.
            An integer can be given to specify the type of DCT (1, 2, 3 or 4).
            if dct=True, the DCT type 2 will be performed, following scipy's convention.
        :param dst: used to perform a Direct Sine Transform (DST) aka a R2R transform.
            An integer can be given to specify the type of DST (1, 2, 3 or 4).
            if dst=True, the DST type 2 will be performed, following scipy's convention.
        :param axes: a list or tuple of axes along which the transform should be made.
            if None, the transform is done along the ndim fastest axes, or all
            axes if ndim is None. For R2C transforms, the fast axis must be
            transformed.
        :param strides: the array strides - needed if not C-ordered.
        :param r2c_odd: this should be set to True to perform an inplace r2c/c2r
            transform with an odd-sized fast (x) axis.
            Explanation: to perform a 1D inplace transform of an array with 100
            elements, the input array should have a 100+2 size, resulting in
            a half-Hermitian array of size 51. If the input data has a size
            of 101, the input array should also be padded to 102 (101+1), and
            the resulting half-Hermitian array also has a size of 51. A
            flag is thus needed to differentiate the cases of 100+2 or 101+1.

        :raises RuntimeError:  if the transform dimensions or data type
            are not allowed by VkFFT.
        """
        self.app = None
        self.config = None
        if ((dct or dst) and r2c) or (dct and dst):
            raise RuntimeError("R2C, DCT and DST are mutually exclusive")
        if (r2c or dct or dst) and dtype not in [np.float16, np.float32, np.float64]:
            raise RuntimeError("R2C, DST or DCT selected but input type is not real")
        self.fast_axis = len(shape) - 1  # default for C-ordered if strides is None
        if strides is not None:
            # Getting the real fast axis can be tricky. Lots of corner cases,
            # as the stride can be zero for axes of size 1...
            if axes is not None and np.sum([sh > 1 for sh in shape]) == 1:
                # Strides won't help, so use the last transformed axis listed
                # as fast axis-following numpy convention
                self.fast_axis = axes[-1]
            else:
                s0 = strides_nonzero(strides)
                if not np.all([s0[i] >= s0[i + 1] for i in range(len(shape) - 1)]):
                    # F-ordered array
                    self.fast_axis = 0
        if r2c and axes is not None:
            if self.fast_axis not in axes and -len(shape) + self.fast_axis not in axes:
                raise RuntimeError(f"the fast axis must be transformed for R2C"
                                   f" [axes={axes}, strides={strides}], "
                                   f"fast axis={self.fast_axis}/{-len(shape) + self.fast_axis}]")
        if r2c and inplace:
            if shape[self.fast_axis] % 2:
                raise RuntimeError(f"For an inplace R2C/C2R transform, the supplied array shape {shape} "
                                   f"must be even along the fast (x) axis. If the transform size nx is "
                                   f"even, two buffer elements should be added to the end of the axis."
                                   f"If it is odd, only one element should be added. In both cases, "
                                   f"the complex half-Hermitian array size will be nx//2+1.")
        # Get the final shape passed to VkFFT, collapsing non-transform axes
        # as necessary. The calculated shape has 4 dimensions (nx, ny, nz, n_batch).
        self.shape, self.n_batch, self.skip_axis, self.ndim, self.axes0 = \
            calc_transform_axes(shape, axes, ndim, strides)
        # original shape (without collapsed non-transformed axes)
        self.shape0 = shape
        self.strides0 = strides
        self.inplace = inplace
        self.r2c = r2c
        self.r2c_odd = r2c_odd
        if dct is False:
            self.dct = 0
        elif dct is True:
            self.dct = 2
        else:
            self.dct = dct
        if dst is False:
            self.dst = 0
        elif dst is True:
            self.dst = 2
        else:
            self.dst = dst
        if dct and self.dct < 1 or self.dct > 4:
            raise RuntimeError("Only DCT of types 1, 2, 3 and 4 are allowed")
        if dst and self.dst < 1 or self.dst > 4:
            raise RuntimeError("Only DST of types 1, 2, 3 and 4 are allowed")

        # These parameters will be filled in by the different backends
        # Size of the temp buffer allocated by VkFFT
        self.tmp_buffer_nbytes = 0
        # 0 or 1 for each axis, only if the Bluestein algorithm is used (same length as self.shape)
        self.use_bluestein_fft = None
        # number of axis upload per dimension (same length as self.shape)
        self.nb_axis_upload = None

        # Experimental parameters. Not much difference is seen, so don't document this,
        # VkFFT default parameters seem fine.

        # force callback version of R2C and R2R (DCT/DST) algorithms for all usecases (0 - off, 1 - on)
        # this is normally activated automatically by VkFFT for odd sizes.
        if "forceCallbackVersionRealTransforms" in kwargs:
            self.forceCallbackVersionRealTransforms = kwargs["forceCallbackVersionRealTransforms"]
        else:
            self.forceCallbackVersionRealTransforms = -1

        # disables unshuffling of Four step algorithm. Requires tempbuffer allocation (0 - off, 1 - on)
        if "disableReorderFourStep" in kwargs:
            self.disableReorderFourStep = kwargs["disableReorderFourStep"]
        else:
            self.disableReorderFourStep = -1

        # uint64_t coalescedMemory - number of bytes to coalesce per one transaction.
        # For Nvidia and AMD is equal to 32, Intel is equal to 64. Going to work regardless,
        # but if specified by the user correctly, the performance will be higher. Default 64
        # for other GPUs. For half-precision should be multiplied by two. Should be a power of two.
        if "coalescedMemory" in kwargs:
            self.coalescedMemory = kwargs["coalescedMemory"]
        else:
            self.coalescedMemory = -1

        # uint64_t numSharedBanks - configure the number of shared banks on the target GPU.
        # Default 32. Minor performance boost as it solves shared memory conflicts for
        # the power of two systems.
        if "numSharedBanks" in kwargs:
            self.numSharedBanks = kwargs["numSharedBanks"]
        else:
            self.numSharedBanks = -1

        # uint64_t aimThreads - try to aim all kernels at this amount of threads.
        # Gains/losses are not predictable, just a parameter to play with
        # (it is not guaranteed that the target kernel will use that many threads).
        # Default 128.
        if "aimThreads" in kwargs:
            self.aimThreads = kwargs["aimThreads"]
        else:
            self.aimThreads = -1

        # uint64_t performBandwidthBoost - try to reduce coalesced number by a factor of X
        # to get bigger sequence in one upload for strided axes.
        # Default: -1(inf) for DCT, 2 for Bluestein’s algorithm (or -1 if DCT), 0 otherwise
        if "performBandwidthBoost" in kwargs:
            self.performBandwidthBoost = kwargs["performBandwidthBoost"]
        else:
            self.performBandwidthBoost = -1

        # uint64_t registerBoost - specify if the register file size is bigger than
        # shared memory and can be used to extend it X times (on Nvidia 256KB register
        # file can be used instead of 32KB of shared memory, set this constant to 4 to
        # emulate 128KB of shared memory). Default 1 - no over-utilization. In Vulkan,
        # OpenCL and Level Zero it is set to 4 on Nvidia GPUs, to 2 if the driver
        # shows 64KB or more of shared memory on AMD, to 2 if the driver shows less
        # than 64KB of shared memory on AMD, to 1 if the driver shows 64KB or more
        # of shared memory on Intel, to 2 if the driver shows less than 64KB of shared
        # memory on Intel.
        if "registerBoost" in kwargs:
            self.registerBoost = kwargs["registerBoost"]
        else:
            self.registerBoost = -1

        # uint64_t registerBoostNonPow2 - specify if register over-utilization should
        # be used on non-power of 2 sequences. Default 0, set to 1 to enable.
        if "registerBoostNonPow2" in kwargs:
            self.registerBoostNonPow2 = kwargs["registerBoostNonPow2"]
        else:
            self.registerBoostNonPow2 = -1

        # uint64_t registerBoost4Step - specify if register file over-utilization
        # should be used in big sequences (>2^14), same definition as registerBoost.
        # Default 1.
        if "registerBoost4Step" in kwargs:
            self.registerBoost4Step = kwargs["registerBoost4Step"]
        else:
            self.registerBoost4Step = -1

        # # uint64_t maxComputeWorkGroupCount[3] - how many workgroups can be launched
        # # at one dispatch. Automatically derived from the driver, can be artificially
        # # lowered. Then VkFFT will perform a logical split and extension of the
        # # number of workgroups to cover the required range.
        # if "maxComputeWorkGroupCount" in kwargs:
        #     self.maxComputeWorkGroupCount = kwargs["maxComputeWorkGroupCount"]
        # else:
        #     self.maxComputeWorkGroupCount = (-1, -1, -1)
        #
        # # uint64_t maxComputeWorkGroupSize[3] - max dimensions of the workgroup.
        # # Automatically derived from the driver. Can be modified if there are
        # # some issues with the driver (as there were with ROCm 4.0, when it returned
        # # 1024 for maxComputeWorkGroupSize and actually supported only up to 256 threads).
        # if "maxComputeWorkGroupSize" in kwargs:
        #     self.maxComputeWorkGroupSize = kwargs["maxComputeWorkGroupSize"]
        # else:
        #     self.maxComputeWorkGroupSize = (-1, -1, -1)
        #
        # # uint64_t maxThreadsNum - max number of threads per block. Similar to maxCompute
        # # - WorkGroupSize, but aggregated. Automatically derived from the driver.
        # if "maxThreadsNum" in kwargs:
        #     self.maxThreadsNum = kwargs["maxThreadsNum"]
        # else:
        #     self.maxThreadsNum = -1
        #
        # # uint64_t sharedMemorySizeStatic - available for static allocation shared
        # # memory size, in bytes. Automatically derived from the driver.
        # if "sharedMemorySizeStatic" in kwargs:
        #     self.sharedMemorySizeStatic = kwargs["sharedMemorySizeStatic"]
        # else:
        #     self.sharedMemorySizeStatic = -1
        #
        # # uint64_t sharedMemorySize - available for allocation shared memory size,
        # # in bytes. VkFFT uses dynamic shared memory in CUDA/HIP as it allows for
        # # bigger allocations.
        # if "sharedMemorySize" in kwargs:
        #     self.sharedMemorySize = kwargs["sharedMemorySize"]
        # else:
        #     self.sharedMemorySize = -1
        #
        # # uint64_t sharedMemorySizePow2 - the power of 2 which is less or equal to
        # # sharedMemorySize, in bytes.
        # if "sharedMemorySizePow2" in kwargs:
        #     self.sharedMemorySizePow2 = kwargs["sharedMemorySizePow2"]
        # else:
        #     self.sharedMemorySizePow2 = -1

        # uint64_t warpSize - number of threads per warp/wavefront. Automatically derived
        # from the driver, but can be modified (can increase performance, though
        # unpredictable as defaults have good values). Must be a power of two.
        if "warpSize" in kwargs:
            self.warpSize = kwargs["warpSize"]
        else:
            self.warpSize = -1

        self.groupedBatch = [-1, -1, -1]
        if "groupedBatch" in kwargs:
            self.groupedBatch = list(kwargs["groupedBatch"])
            # In case less than 3 parameters where given
            self.groupedBatch[:len(kwargs["groupedBatch"])] = kwargs["groupedBatch"]

        if "useLUT" in kwargs:
            # useLUT=1 may be beneficial on platforms which have a low accuracy for
            # the native sincos functions.
            if kwargs["useLUT"] is None:
                self.use_lut = -1
            else:
                self.use_lut = kwargs["useLUT"]
        elif config.USE_LUT is not None:
            self.use_lut = config.USE_LUT
        else:
            self.use_lut = -1

        if "keepShaderCode" in kwargs:
            # This will print the compiled code if equal to 1
            self.keepShaderCode = kwargs["keepShaderCode"]
        else:
            self.keepShaderCode = -1

        if norm == "backward":
            norm = 1
        self.norm = norm

        # Precision: number of bytes per float
        if dtype in [np.float16, complex32]:
            self.precision = 2
        elif dtype in [np.float32, np.complex64]:
            self.precision = 4
        elif dtype in [np.float64, np.complex128]:
            self.precision = 8

    def __str__(self):
        """
        Get a string describing the VkFFTApp properties, e.g.:
          VkFFTApp[OpenCL]:(212,212+2)     R2C/s/i [RR] [11] buf=    0
        This includes VkFFTApp with the backend (OpenCL or CUDA) used, followed
        by the shape of the array (in python order),
        then the type of transform (C2C or R2C or DCT/DST), then
        h/s/d for half/single/double precisions,
        i or o for in or out-of-place transforms,
        [???] with a letter indicating for each axis if it uses a
        [r]adix, [B]luestein or [R]ader transform.
        Then [nnn] indicates how manu uploads are used per axis,
        and finally the size of the temporary buffer allocated by VkFFT
        is indicated, if any.
        """
        bufs = self.get_tmp_buffer_str()
        ft_type = self.get_algo_str()

        sh = self.shape.copy()
        sh[-1] *= self.n_batch
        naxup = ''.join([str(i) for i in self.get_nb_upload()])
        s = "VkFFTApp[OpenCL]:" if 'opencl' in str(self.__class__) else "VkFFTApp[CUDA]:  "
        s += f"{self.get_shape_str():15s}"
        if self.dct:
            s += f"DCT{self.dct}"
        elif self.dst:
            s += f"DST{self.dst}"
        elif self.r2c:
            s += " R2C"
        else:
            s += " C2C"
        s += {2: "/h", 4: "/s", 8: "/d"}[self.precision]
        s += '/i' if self.inplace else '/o'
        s += f" [{ft_type}] [{naxup}]"
        s += f" buf={bufs}"
        return s

    def _get_fft_scale(self, norm):
        """Return the scale factor by which an array must be multiplied to keep its L2 norm
        after a forward FT
        :param norm: the norm option for which the scale is computed, either 0 or 1
        :return: the scale factor, as a numpy float with the precision used for the fft
        """
        dtype = np.float32
        if self.precision == 8:
            dtype = np.float64
        elif self.precision == 2:
            dtype = np.float16
        s = 1
        ndim_real = 0
        for i in range(self.ndim):
            if not self.skip_axis[i]:
                s *= self.shape[i]
                ndim_real += 1
        s = np.sqrt(s)
        if self.r2c and self.inplace:
            # Note: this is still correct for non-C-ordered arrays since
            # self.shape has been re-ordered
            if self.r2c_odd:
                s *= np.sqrt((self.shape[0] - 1) / self.shape[0])
            else:
                s *= np.sqrt((self.shape[0] - 2) / self.shape[0])
        if self.dct or self.dst:
            s *= 2 ** (0.5 * ndim_real)
            if max(self.dct, self.dst) != 4:  # if dct is used, dst=0 and inversely
                warnings.warn("A DST or DCT type 2 or 3 cannot be strictly normalised, using approximation,"
                              " see https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II")
        if norm == 0 or norm == 1:
            return dtype(1 / s)
        elif norm == "ortho":
            return dtype(1)
        raise RuntimeError("Unknown norm choice !")

    def get_fft_scale(self):
        """Return the scale factor by which an array must be multiplied to keep its L2 norm
        after a forward FT
        """
        return self._get_fft_scale(self.norm)

    def _get_ifft_scale(self, norm):
        """Return the scale factor by which an array must be multiplied to keep its L2 norm
        after a backward FT
        :param norm: the norm option for which the scale is computed, either 0 or 1
        :return: the scale factor, as a numpy float with the precision used for the fft
        """
        dtype = np.float32
        if self.precision == 8:
            dtype = np.float64
        elif self.precision == 2:
            dtype = np.float16
        s = 1
        s_dct = 1  # used also for dst
        for i in range(self.ndim):
            if not self.skip_axis[i]:
                s *= self.shape[i]
                if self.dct or self.dst:
                    s_dct *= np.sqrt(2)
        s = np.sqrt(s)
        if self.r2c and self.inplace:
            if self.r2c_odd:
                s *= np.sqrt((self.shape[0] - 1) / self.shape[0])
            else:
                s *= np.sqrt((self.shape[0] - 2) / self.shape[0])
        r2r_type = max(self.dct, self.dst)
        if (self.dct or self.dst) and r2r_type != 4:
            warnings.warn("A DST or DCT type 2 or 3 cannot be strictly normalised, using approximation,"
                          " see https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II")
        if norm == 0:
            return dtype(1 / (s * s_dct))
        elif norm == 1:
            # Not sure why the difference in scale factors
            if r2r_type == 2:
                s_dct = s_dct ** 1
            elif r2r_type == 3:
                s_dct = s_dct ** 2
            elif r2r_type == 4:
                s_dct = s_dct ** 3
            return dtype(s * s_dct)
        elif norm == "ortho":
            return dtype(1)
        raise RuntimeError("Unknown norm choice !")

    def get_ifft_scale(self):
        """Return the scale factor by which an array must be multiplied to keep its L2 norm
        after a backward FT
        """
        return self._get_ifft_scale(self.norm)

    def get_tmp_buffer_nbytes(self):
        """
        Return the size (in bytes) of the temporary buffer allocated
        by VkFFT for the transform, if any.
        """
        return self.tmp_buffer_nbytes

    def get_tmp_buffer_str(self):
        """
        Get a string with the size of the temporary buffer allocated
        by VkFFT, e.g. '0', '123kB', '1.2GB', etc. Uses 6 chars.
        """
        b = self.tmp_buffer_nbytes
        return "    0  " if b == 0 else f"{b / 1024 ** 3:5.1f}GB" if b >= 1000 * 1024 ** 2 \
            else f"{b / 1024 ** 2:5.1f}MB" if b >= 1000 * 1024 \
            else f"{b / 1024 :5.1f}kB" if b >= 1000 else f"{b:6d}B "

    def get_algo_str(self, vkfft_axes=False):
        """
        Return a string indicating the type of algorithm used for each axis,
        either [r]adix, [B]luestein or [R]ader, or '-' if the axis
        is skipped.
        """
        if vkfft_axes:
            tmp = ''
            for i in range(len(self.shape)):
                if self.skip_axis[i]:
                    tmp += '-'
                elif self.is_radix_transform(i, vkfft_axes=True):
                    tmp += 'r'
                elif self.is_rader_transform(i, vkfft_axes=True):
                    tmp += 'R'
                else:
                    tmp += 'B'
            return tmp
        else:
            tmp = ''
            for i in range(len(self.shape0)):
                if self.skip_axis[self._get_vkfft_axes(i)]:
                    tmp += '-'
                elif self.is_radix_transform(i):
                    tmp += 'r'
                elif self.is_rader_transform(i):
                    tmp += 'R'
                else:
                    tmp += 'B'
            return tmp

    def get_shape_str(self, vkfft_axes=False):
        """
        Get a string with the shape of the array, including the +1
        or +2 for inplace r2c transforms.

        :param vkfft_axes: True to use the index relative to VkFFT
            axes, False (the default) otherwise.
            See _get_vkfft_axes() for details.
        """
        # TODO: handle non C-contiguous array (strides) for inplace r2c
        if vkfft_axes:
            sh = list(self.shape).copy()
        else:
            sh = list(self.shape0).copy()

        if self.r2c and self.inplace:
            # Need to figure out the fast axis for the +1 or +2
            if vkfft_axes:
                fast_axis = 0
            else:
                # _get_vkfft_axes() takes into account strides
                fast_axis = np.argmin(self._get_vkfft_axes())

            if self.r2c_odd:
                r2c_inplace_pad = 1
            else:
                r2c_inplace_pad = 2

            sh[fast_axis] -= r2c_inplace_pad
            tmp = [str(sh[i]) + f'+{r2c_inplace_pad}' if i == fast_axis
                   else str(sh[i]) for i in range(len(sh))]
            return f"({','.join(tmp)})"
        return f"({','.join([str(n) for n in sh])})"

    def _get_vkfft_axes(self, i0=None):
        """
        Get the index of an axis in self.shape, self.skip_axis,
        self.use_bluestein_fft, etc..., from the original index.
        This is used because consecutive non-transformed axes are collapsed (merged).
        Additionally, VkFFT indexes the dimensions
        as [nx ny nz] rather than [nz ny nx], which is also taken care of.
        Finally, this takes into account strides if the array is not C-contiguous,
        which will change the order of the axes stored for VkFFT.
        Example: an array of shape (10,8,6,4) and axes=[-4,-1] will be internally
        seen as an array of shape (4,48,10).

        :param i0: the index in the original numpy array shape (can be >=0 or <0)
        :return: the index used for the same axis with the VkFFT order, i.e.
            reversed and after collapsing non-transformed axes. If i0=None,
            the index of all dimensions is returned as a list
        """
        n = len(self.shape0)
        i1 = len(self.shape) - 1
        idx = [i1]
        for i in range(n - 1):
            if i in self.axes0 or i - n in self.axes0 or i + 1 in self.axes0 or i + 1 - n in self.axes0:
                i1 -= 1
            idx.append(i1)
        if self.strides0 is not None:
            idx = np.take(idx, np.argsort(self.strides0)[::-1])
        if i0 is None:
            return idx
        return idx[i0]

    def is_bluestein_transform(self, axis=None, vkfft_axes=False):
        """
        Return True if the transform used along a given axis uses
        Bluestein's algorithm, or False

        :param axis: the index of one axis. If None, a list of values
            for all the axis is returned
        :param vkfft_axes: True to use the index relative to VkFFT
            axes, False (the default) otherwise.
            See _get_vkfft_axes() for details.
        """
        b = self.use_bluestein_fft
        if vkfft_axes:
            if axis is not None:
                return bool(b[axis])
            else:
                return b
        else:
            if axis is not None:
                return b[self._get_vkfft_axes(axis)]
            else:
                return [b[self._get_vkfft_axes(i)] for i in range(len(self.shape0))]

    def is_rader_transform(self, axis=None, vkfft_axes=False):
        """
        Return True if the transform used along a given axis uses
        Rader's algorithm, or False

        :param axis: the index of one axis. If None, a list of values
            for all the axis is returned
        :param vkfft_axes: True to use the index relative to VkFFT
            axes, False (the default) otherwise.
            See _get_vkfft_axes() for details.
        """
        b = self.use_bluestein_fft
        t = self.skip_axis
        sh = list(self.shape).copy()
        if self.r2c and self.inplace:
            # Fast axis is always first in the VkFFT order
            if self.r2c_odd:
                sh[0] -= 1
            else:
                sh[0] -= 2

        if self.dct == 1 or self.dst == 1:
            # Transforms are mapped to a C2C of size 2N-2
            sh = [max(2 * n - 2, 1) for n in sh]  # avoid 2*(n=1)-2=0
        elif self.dct == 4 or self.dst == 4:
            # even mapped to C2C of half-size, else C2C of same size
            sh = [n if n % 2 else n // 2 for n in sh]
        r = [max(primes(n)) <= 13 for n in sh]

        if vkfft_axes:
            if axis is not None:
                return not (b[axis] or t[axis] or r[axis])
            else:
                return [not (b[i] or t[i] or r[i]) for i in range(len(b))]
        else:
            if axis is not None:
                i = self._get_vkfft_axes(axis)
                return not (b[i] or t[i] or r[i])
            else:
                return [not (b[self._get_vkfft_axes(i)] or
                             t[self._get_vkfft_axes(i)] or
                             r[self._get_vkfft_axes(i)]) for i in range(len(self.shape0))]

    def is_radix_transform(self, axis=None, vkfft_axes=False):
        """
        Return True if the transform used along a given axis uses
        a radix algorithm, or False

        :param axis: the index of one axis. If None, a list of values
            for all axes is returned
        :param vkfft_axes: True to use the index relative to VkFFT
            axes, False (the default) otherwise.
            See _get_vkfft_axes() for details.
        """
        t = self.skip_axis
        sh = list(self.shape).copy()
        if self.r2c and self.inplace:
            # Fast axis is always first in the VkFFT order
            if self.r2c_odd:
                sh[0] -= 1
            else:
                sh[0] -= 2

        if self.dct == 1 or self.dst == 1:
            # Transforms are mapped to a C2C of size 2N-2
            sh = [max(2 * n - 2, 1) for n in sh]  # avoid 2*(n=1)-2=0
        elif self.dct == 4 or self.dst == 4:
            # even mapped to C2C of half-size, else C2C of same size
            sh = [n if n % 2 else n // 2 for n in sh]
        r = [max(primes(n)) > 13 for n in sh]

        if vkfft_axes:
            if axis is not None:
                return not (t[axis] or r[axis])
            else:
                return [not (t[i] or r[i]) for i in range(len(t))]
        else:
            if axis is not None:
                i = self._get_vkfft_axes(axis)
                return not (t[i] or r[i])
            else:
                return [not (t[self._get_vkfft_axes(i)] or r[self._get_vkfft_axes(i)])
                        for i in range(len(self.shape0))]

    def get_nb_upload(self, axis=None, vkfft_axes=False):
        """
        Number of uploads for the transform along given axes - ideally 1
        so that each transform corresponds to 1 read and 1 write of the array.

        :param axis: the index of one axis. If None, a list of values
            for all axes is returned
        :param collapsed_axes: True to use the index relative to collapsed
            axes, False (the default) otherwise.
            See _get_vkfft_axes() for details.
        """
        n = self.nb_axis_upload
        if vkfft_axes:
            if axis is not None:
                return n[::-1][axis]
            else:
                return n[::-1]
        else:
            if axis is not None:
                i = self._get_vkfft_axes(axis)
                return n[i]
            else:
                return [n[self._get_vkfft_axes(i)] for i in range(len(self.shape0))]
