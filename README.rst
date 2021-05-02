pyvkfft - python interface to the CUDA and OpenCL backends of VkFFT (Vulkan Fast Fourier Transform library)
===========================================================================================================

`VkFFT <https://github.com/DTolm/VkFFT>`_ is a GPU-accelerated Fast Fourier Transform library
for Vulkan/CUDA/HIP/OpenCL.

pyvkfft offers a simple python interface to the **CUDA** and **OpenCL** backends of VkFFT, compatible with pyCUDA and pyOpenCL.

*The code is now in a working state, and passes all unit tests ; no errors are reported by either valgrind or cuda-memcheck.*

Installation
------------

Install using `pip install pyvkfft` (works on macOS and Linux).

Note that the PyPI archive includes `vkfft.h` and will automatically install `pyopencl`,
as well as `pycuda` if a CUDA environment is detected.

Requirements:

- `vkfft.h` installed in the usual include directories, or in the 'src' directory
- `pyopencl` and the opencl libraries/development tools for the opencl backend
- `pycuda` and CUDA developments tools (`nvcc`) for the cuda backend (optional)
- `numpy`

This package can be installed from source using `python setup.py install`.

Examples
--------
See the script and notebook in the examples directory.
The notebook is also `available on google colab
<https://colab.research.google.com/drive/1YJKtIwM3ZwyXnMZfgFVcpbX7H-h02Iej?usp=sharing>`_.
Make sure to select a GPU for the runtime. This may fail on old architectures (Kepler- to be confirmed)


Status
------
What works:

- CUDA and OpenCL backends
- C2C, R2C/C2R for inplace and out-of-place transforms
- single and double precision for all transforms
- 1D, 2D and 3D transforms (always performed from the last axes).
- 1D and 2D FFT accept arrays of any number of dimensions (batch transforms), 3D FFT
  only accepts 3D arrays.
- allowed prime factors (radix) of the transform axes are 2, 3, 5, 7, 11 and 13
- normalisation=0 (array L2 norm * array size on each transform) and 1 (the backward
  transform divides the L2 norm by the array size, so FFT*iFFT restores the original array)
- now testing the FFT size does not exceed the allowed maximum prime number decomposition (13)
- unit tests for all transforms: see test sub-directory.
- Note that out-of-place C2R transform currently destroys the complex array for FFT dimensions >=2
- tested on macOS (10.13.6) and Linux.
- inplace transforms do not require an extra buffer or work area (as in cuFFT), unless the x
  size is larger than 8192, or if the y and z FFT size are larger than 2048. In that case
  a buffer of a size equal to the array is necessary. This makes larger FFT transforms possible
  based on memory requiremnts (even for R2C !) compared to cuFFT. For example you can compute
  the 3D FFT for a 1600**3 complex64 array withon 32GB of memory.

Performance
-----------
See the benchmark notebook, which allows to plot OpenCL and CUDA backend throughput, as well as compare
with cuFFT (using scikit-cuda) and clFFT (using gpyfft).

Example result for 2D FFT with array dimensions of 16xNxN using a Titan V:

.. image:: https://raw.githubusercontent.com/vincefn/pyvkfft/master/doc/benchmark-2DFFT-TITAN_V-Linux.png

Note that in this plot the computed throughput is theoretical, as if each transform axis for the
couple (FFT, iFFT) required exactly one read and one write. This is obviously not true,
and explains the drop after N=1024 for cuFFT and (in a smaller extent) vkFFT.

The general results are:

* vkFFT throughput is similar to cuFFT up to N=150, then slightly lower up to N=1024. For N>1024
  vkFFT is much more efficient than cuFFT due to the smaller number of read and write per FFT axis
  (apart from isolated power-of-2 or power-of-3 sizes)
* the OpenCL and CUDA backends of vkFFT perform similarly, as expected. [Note that this should
  be true *as long as the card is only used for computing*. If it is also used for display,
  then performance may be different, e.g. for nvidia cards opencl performance is more affected
  when being used for display than the cuda backend]
* clFFT (via gpyfft) generally performs much worse than the other transforms, though this was
  tested using nVidia cards. (Note that the clFFT/gpyfft benchmark tries all FFT axis permutations
  to find the fastest combination)

TODO
----

- access to the other backends:

  - for vulkan and rocm this only makes sense combined to a pycuda/cupy/pyopencl equivalent.
- support cupy arrays (this probably requires little change so a cupy user/developer contribution is welcome)
- out-of-place C2R transform without modifying the C array ? This would require using a R
  array padded with two wolumns, as for the inplace transform
- windows support (contribution welcome to setup.py)
- half precision ?
- convolution ?
- zero-padding ?
- access to tweaking parameters in VkFFTConfiguration ?
- access to the code of the generated kernels ?
