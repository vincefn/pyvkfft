# -*- coding: utf-8 -*-

# PyVkFFT
#   (c) 2021- : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

import ctypes
import numpy as np
from .tune import tune_vkfft

try:
    import pycuda.driver as cu_drv

    has_pycuda = True
except ImportError:
    has_pycuda = False
try:
    import cupy as cp

    has_cupy = True
except ImportError:
    has_cupy = False
    import sys

    if has_pycuda is False and 'sphinx' not in sys.modules:
        raise ImportError("You need either PyCUDA or CuPy to use pyvkfft.cuda.")

from .base import load_library, VkFFTApp as VkFFTAppBase, check_vkfft_result, ctype_int_size_p

try:
    _vkfft_cuda = load_library("_vkfft_cuda")


    class _types:
        """Aliases"""
        vkfft_config = ctypes.c_void_p
        stream = ctypes.c_void_p
        vkfft_app = ctypes.c_void_p


    _vkfft_cuda.make_config.restype = ctypes.c_void_p
    _vkfft_cuda.make_config.argtypes = [ctype_int_size_p, ctypes.c_size_t,
                                        ctypes.c_void_p, ctypes.c_void_p, _types.stream,
                                        ctypes.c_int, ctypes.c_size_t, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_size_t,
                                        ctype_int_size_p, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_int, ctype_int_size_p, ctypes.c_int]

    _vkfft_cuda.init_app.restype = ctypes.c_void_p
    _vkfft_cuda.init_app.argtypes = [_types.vkfft_config, ctypes.POINTER(ctypes.c_int),
                                     ctypes.POINTER(ctypes.c_size_t),
                                     ctype_int_size_p, ctype_int_size_p]

    _vkfft_cuda.fft.restype = ctypes.c_int
    _vkfft_cuda.fft.argtypes = [_types.vkfft_app, ctypes.c_void_p, ctypes.c_void_p]

    _vkfft_cuda.ifft.restype = ctypes.c_int
    _vkfft_cuda.ifft.argtypes = [_types.vkfft_app, ctypes.c_void_p, ctypes.c_void_p]

    _vkfft_cuda.free_app.restype = None
    _vkfft_cuda.free_app.argtypes = [_types.vkfft_app]

    _vkfft_cuda.free_config.restype = None
    _vkfft_cuda.free_config.argtypes = [_types.vkfft_config]

    _vkfft_cuda.vkfft_max_fft_dimensions.restype = ctypes.c_uint32
    _vkfft_cuda.vkfft_max_fft_dimensions.argtypes = None
except OSError:
    # This is used for doc generation
    import sys

    if 'sphinx' in sys.modules:
        pass
    else:
        raise


class VkFFTApp(VkFFTAppBase):
    """
    VkFFT application interface, similar to a cuFFT plan.
    """

    def __init__(self, shape, dtype: type, ndim=None, inplace=True, stream=None, norm=1,
                 r2c=False, dct=False, dst=False, axes=None, strides=None, tune_config=None,
                 r2c_odd=False, verbose=False, **kwargs):
        """

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
            float array should have a shape of (..., nx+2) if nx is even
            or (..., nx+1) if nx is odd, the last one or two columns
            being ignored in the input data, and the resulting
            complex array (using pycuda's GPUArray.view(dtype=np.complex64) to
            reinterpret the type) will have a shape (..., nx//2 + 1).
            For an out-of-place transform, if the input (real) shape is (..., nx),
            the output (complex) shape should be (..., nx//2+1).
            Note that for C2R transforms with ndim>=2, the source (complex) array
            is modified.
            For an inplace transform with an odd-sized x-axis, see the r2c_odd
            parameter.
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
        :param tune_config: this can be used to automatically generate an
            optimised set of VkFFT parameters by testing various configurations
            and measuring the FFT speed, in a manner similar to fftw's FFTW_MEASURE.
            This should be a dictionary including the backend used and the parameter
            values which will be tested.
            This is EXPERIMENTAL, as wrong parameters may lead to crashes.
            Note that this will allocate temporary GPU arrays, unless the arrays
            to used have been passed as parameters ('dest' and 'src').
            Examples:
            tune={'backend':'cupy} - minimal example, will automatically test a small
            set of parameters (4 to 10 tests). Recommended !
            tune={'backend':'cupy, 'warpSize':[8,16,32,64,128]}: this will test
            5 possible values for the warpSize.
            tune={'backend':'cupy, 'groupedBatch':[[-1,-1,-1],[8,8,8], [4,16,16}:
            this will test 3 possible values for groupedBatch. This one is more
            tricky to use.
            tune={'backend':'cupy, 'warpSize':[8,16,32,64,128], 'src':a}: this
            will test 5 possible values for the warpSize, with a given source GPU
            array. This would only be valid for an inplace transform as no
            destination array is given.
        :param r2c_odd: this should be set to True to perform an inplace r2c/c2r
            transform with an odd-sized fast (x) axis.
            Explanation: to perform a 1D inplace transform of an array with 100
                elements, the input array should have a 100+2 size, resulting in
                a half-Hermitian array of size 51. If the input data has a size
                of 101, the input array should also be padded to 102 (101+1), and
                the resulting half-Hermitian array also has a size of 51. A
                flag is thus needed to differentiate the cases of 100+2 or 101+1.
        :param verbose: if True, print a 1-string info about this VkFFTApp.
            See __str__ for details.

        :raises RuntimeError: if the initialisation fails, e.g. if the CUDA
            driver has not been properly initialised, or if the transform dimensions
            are not allowed by VkFFT.
        """
        if tune_config is not None:
            kwargs = tune_vkfft(tune_config, shape=shape, dtype=dtype, ndim=ndim, inplace=inplace, stream=stream,
                                norm=norm, r2c=r2c, dct=dct, dst=dst, axes=axes, strides=strides, verbose=False,
                                r2c_odd=r2c_odd, **kwargs)[0]
        super().__init__(shape, dtype, ndim=ndim, inplace=inplace, norm=norm, r2c=r2c,
                         dct=dct, dst=dst, axes=axes, strides=strides, r2c_odd=r2c_odd, **kwargs)

        self.stream = stream

        self.config = self._make_config()
        if self.config is None:
            raise RuntimeError("Error creating VkFFTConfiguration. Was the CUDA context properly initialised ?")

        res = ctypes.c_int(0)
        # Size of tmp buffer allocated by VkFFT - if any
        tmp_buffer_nbytes = ctypes.c_size_t(0)
        # 0 or 1 for each axis, only if the Bluestein algorithm is used
        use_bluestein_fft = np.zeros(vkfft_max_fft_dimensions(), dtype=int)
        # number of axis upload per dimension
        num_axis_upload = np.zeros(vkfft_max_fft_dimensions(), dtype=int)

        self.app = _vkfft_cuda.init_app(self.config, ctypes.byref(res),
                                        ctypes.byref(tmp_buffer_nbytes),
                                        use_bluestein_fft, num_axis_upload)

        check_vkfft_result(res, shape, dtype, ndim, inplace, norm, r2c, dct, dst, axes, "cuda")

        if self.app is None:
            raise RuntimeError("Error creating VkFFTApplication. Was the CUDA driver initialised ?")
        if has_pycuda:
            # TODO: This is a kludge to keep a reference to the context, so that it is deleted
            #  after the app in __delete__, which throws an error if the context does not exist
            #  anymore. Except that we cannot be sure this is the right context, if a stream
            #  has been given because we don't have access to cuStreamGetCtx from python...
            self._ctx = cu_drv.Context.get_current()

        self.tmp_buffer_nbytes = np.int64(tmp_buffer_nbytes)
        self.use_bluestein_fft = [bool(n) for n in use_bluestein_fft[:len(self.shape)]]
        self.nb_axis_upload = [int(num_axis_upload[i] * (self.skip_axis[i] is False))
                               for i in range(len(self.shape))]

        if verbose:
            print(self)

    def __del__(self):
        """ Takes care of deleting allocated memory in the underlying
        VkFFTApplication and VkFFTConfiguration.
        """
        if self.app is not None:
            _vkfft_cuda.free_app(self.app)
        if self.config is not None:
            _vkfft_cuda.free_config(self.config)

    def _make_config(self):
        """ Create a vkfft configuration for a FFT transform"""
        if len(self.shape) > vkfft_max_fft_dimensions():
            raise RuntimeError(f"Too many FFT dimensions after collapsing non-transform axes: "
                               f"{len(self.shape)}>{vkfft_max_fft_dimensions()}")

        shape = np.ones(vkfft_max_fft_dimensions(), dtype=int)
        shape[:len(self.shape)] = self.shape

        skip = np.zeros(vkfft_max_fft_dimensions(), dtype=int)
        skip[:len(self.skip_axis)] = self.skip_axis

        grouped_batch = np.empty(vkfft_max_fft_dimensions(), dtype=int)
        grouped_batch.fill(-1)
        grouped_batch[:len(self.groupedBatch)] = self.groupedBatch

        if self.r2c and self.inplace:
            # the last one or two columns are ignored in the R array, and will be used
            # in the C array with a size nx//2+1
            if self.r2c_odd:
                shape[0] -= 1
            else:
                shape[0] -= 2

        s = 0
        if self.stream is not None:
            if has_pycuda:
                if isinstance(self.stream, cu_drv.Stream):
                    s = self.stream.handle
            if has_cupy:
                if isinstance(self.stream, cp.cuda.Stream):
                    s = self.stream.ptr
            if s == 0 and isinstance(self.stream, int):
                # Assume the ptr or handle was passed
                s = self.stream

        if self.norm == "ortho":
            norm = 0
        else:
            norm = self.norm

        # We pass fake buffer pointer addresses to VkFFT. The real ones will be
        # given when performing the actual FFT.
        dest_gpudata = 2
        if self.inplace:
            dest_gpudata = 0

        return _vkfft_cuda.make_config(shape, self.ndim, 1, dest_gpudata, s,
                                       norm, self.precision, int(self.r2c),
                                       int(self.dct), int(self.dst),
                                       int(self.disableReorderFourStep), int(self.registerBoost),
                                       int(self.use_lut), int(self.keepShaderCode),
                                       self.n_batch, skip,
                                       int(self.coalescedMemory), int(self.numSharedBanks),
                                       int(self.aimThreads), int(self.performBandwidthBoost),
                                       int(self.registerBoostNonPow2), int(self.registerBoost4Step),
                                       int(self.warpSize), grouped_batch,
                                       int(self.forceCallbackVersionRealTransforms))

    def fft(self, src, dest=None):
        """
        Compute the forward FFT

        :param src: the source pycuda.gpuarray.GPUArray or cupy.ndarray
        :param dest: the destination GPU array. Should be None for an inplace transform
        :raises RuntimeError: in case of a GPU kernel launch error
        :return: the transformed array. For a R2C inplace transform, the complex view of the
            array is returned.
        """
        use_cupy = False
        if has_cupy:
            if isinstance(src, cp.ndarray):
                use_cupy = True
        if use_cupy:
            src_ptr = src.__cuda_array_interface__['data'][0]
        else:
            # Must cast the gpudata to int as it can either be a DeviceAllocation object
            # or an int (e.g. when using a view of another array)
            src_ptr = int(src.gpudata)
        if dest is not None:
            if use_cupy:
                dest_ptr = dest.__cuda_array_interface__['data'][0]
            else:
                dest_ptr = int(dest.gpudata)
        else:
            dest_ptr = src_ptr
        if self.inplace:
            if src_ptr != dest_ptr:
                raise RuntimeError("VkFFTApp.fft: dest is not None but this is an inplace transform")
            res = _vkfft_cuda.fft(self.app, int(src_ptr), int(src_ptr))
            check_vkfft_result(res, src.shape, src.dtype, self.ndim, self.inplace, self.norm, self.r2c,
                               self.dct, self.dst, backend="cuda")
            if self.norm == "ortho":
                src *= self._get_fft_scale(norm=0)
            if self.r2c:
                if src.dtype == np.float32:
                    return src.view(dtype=np.complex64)
                elif src.dtype == np.float64:
                    return src.view(dtype=np.complex128)
            return src
        else:
            if dest is None:
                raise RuntimeError("VkFFTApp.fft: dest is None but this is an out-of-place transform")
            if src_ptr == dest_ptr:
                raise RuntimeError("VkFFTApp.fft: dest and src are identical but this is an out-of-place transform")
            if self.r2c:
                assert (dest.size == src.size // src.shape[self.fast_axis] * (src.shape[self.fast_axis] // 2 + 1))
            res = _vkfft_cuda.fft(self.app, int(src_ptr), int(dest_ptr))
            check_vkfft_result(res, src.shape, src.dtype, self.ndim, self.inplace, self.norm, self.r2c,
                               self.dct, self.dst, backend="cuda")
            if self.norm == "ortho":
                dest *= self._get_fft_scale(norm=0)
            return dest

    def ifft(self, src, dest=None):
        """
        Compute the backward FFT

        :param src: the source pycuda.gpuarray.GPUArray or cupy.ndarray
        :param dest: the destination GPU array. Should be None for an inplace transform
        :raises RuntimeError: in case of a GPU kernel launch error
        :return: the transformed array. For a C2R inplace transform, the float view of the
            array is returned.
        """
        use_cupy = False
        if has_cupy:
            if isinstance(src, cp.ndarray):
                use_cupy = True
        if use_cupy:
            src_ptr = src.__cuda_array_interface__['data'][0]
        else:
            # Must cast the gpudata to int as it can either be a DeviceAllocation object
            # or an int (e.g. when using a view of another array)
            src_ptr = int(src.gpudata)
        if dest is not None:
            if use_cupy:
                dest_ptr = dest.__cuda_array_interface__['data'][0]
            else:
                dest_ptr = int(dest.gpudata)
        else:
            dest_ptr = src_ptr
        if self.inplace:
            if dest is not None:
                if src_ptr != dest_ptr:
                    raise RuntimeError("VkFFTApp.fft: dest!=src but this is an inplace transform")
            res = _vkfft_cuda.ifft(self.app, int(src_ptr), int(src_ptr))
            check_vkfft_result(res, src.shape, src.dtype, self.ndim, self.inplace, self.norm, self.r2c,
                               self.dct, self.dst, backend="cuda")
            if self.norm == "ortho":
                src *= self._get_ifft_scale(norm=0)
            if self.r2c:
                if src.dtype == np.complex64:
                    return src.view(dtype=np.float32)
                elif src.dtype == np.complex128:
                    return src.view(dtype=np.float64)
            return src
        else:
            if dest is None:
                raise RuntimeError("VkFFTApp.ifft: dest is None but this is an out-of-place transform")
            if src_ptr == dest_ptr:
                raise RuntimeError("VkFFTApp.ifft: dest and src are identical but this is an out-of-place transform")
            if self.r2c:
                assert (src.size == dest.size // dest.shape[self.fast_axis] * (dest.shape[self.fast_axis] // 2 + 1))
                # Special case, src and dest buffer sizes are different,
                # VkFFT is configured to go back to the source buffer
                res = _vkfft_cuda.ifft(self.app, int(dest_ptr), int(src_ptr))
            else:
                res = _vkfft_cuda.ifft(self.app, int(src_ptr), int(dest_ptr))
            check_vkfft_result(res, src.shape, src.dtype, self.ndim, self.inplace, self.norm, self.r2c,
                               self.dct, self.dst, backend="cuda")
            if self.norm == "ortho":
                dest *= self._get_ifft_scale(norm=0)
            return dest


def vkfft_version():
    """
    Get VkFFT version
    :return: version as X.Y.Z
    """
    int_ver = _vkfft_cuda.vkfft_version()
    return "%d.%d.%d" % (int_ver // 10000, (int_ver % 10000) // 100, int_ver % 100)


def vkfft_max_fft_dimensions():
    """
    Get the maximum number of dimensions VkFFT can handle. This is
    set at compile time. VkFFT default is 4, pyvkfft sets this to 8.
    Note that consecutive non-transformed are collapsed into a single
    axis, reducing the effective number of dimensions.

    :return: VKFFT_MAX_FFT_DIMENSIONS
    """
    return _vkfft_cuda.vkfft_max_fft_dimensions()


def cuda_runtime_version(raw=False):
    """
    Get CUDA runtime version

    :param raw: if True, return the version as X*1000+Y*10+Z
    :return: version as X.Y.Z
    """
    int_ver = _vkfft_cuda.cuda_runtime_version()
    if raw:
        return raw
    return "%d.%d.%d" % (int_ver // 1000, (int_ver % 1000) // 10, int_ver % 10)


def cuda_driver_version(raw=False):
    """
    Get CUDA driver version

    :param raw: if True, return the version as X*1000+Y*10+Z
    :return: version as X.Y.Z
    """
    int_ver = _vkfft_cuda.cuda_driver_version()
    if raw:
        return raw
    return "%d.%d.%d" % (int_ver // 1000, (int_ver % 1000) // 10, int_ver % 10)


def cuda_compile_version(raw=False):
    """
    Get CUDA version against which pyvkfft was compiled

    :param raw: if True, return the version as X*1000+Y*10+Z
    :return: version as X.Y.Z
    """
    if raw:
        return raw
    int_ver = _vkfft_cuda.cuda_compile_version()
    return "%d.%d.%d" % (int_ver // 1000, (int_ver % 1000) // 10, int_ver % 10)
