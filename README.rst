pyvkfft - python interface to the CUDA and OpenCL backends of VkFFT (Vulkan Fast Fourier Transform library)
===========================================================================================================

`VkFFT <https://github.com/DTolm/VkFFT>`_ is a GPU-accelerated Fast Fourier Transform library
for Vulkan/CUDA/HIP/OpenCL.

pyvkfft offers a simple python interface to the **CUDA** and **OpenCL** backends of VkFFT,
compatible with **pyCUDA**, **CuPy** and **pyOpenCL**.

Installation
------------

Install using ``pip install pyvkfft`` (works on macOS, Linux and Windows).

Note that the PyPI archive includes ``vkfft.h`` and will automatically install ``pyopencl``
if opencl is available. However you should manually install either ``cupy`` or ``pycuda``
to use the cuda backend.

Requirements:

- ``vkfft.h`` installed in the usual include directories, or in the 'src' directory
- ``pyopencl`` and the opencl libraries/development tools for the opencl backend
- ``pycuda`` or ``cupy`` and CUDA developments tools (`nvcc`) for the cuda backend
- ``numpy``
- on Windows, this requires visual studio (c++ tools) and a cuda toolkit installation,
  with either CUDA_PATH or CUDA_HOME environment variable.

This package can be installed from source using ``pip install .``.

*Note:* ``python setup.py install`` is now disabled, to avoid messed up environments
where both methods have been used.

Examples
--------

The simplest way to use pyvkfft is to use the ``pyvkfft.fft`` interface, which will
automatically create the VkFFTApp (the FFT plans) according to the type of GPU
arrays (pycuda, pyopencl or cupy), and also cache these apps:

.. code-block:: python

  import pycuda.autoinit
  import pycuda.gpuarray as cua
  from pyvkfft.fft import fftn
  import numpy as np

  d0 = cua.to_gpu(np.random.uniform(0,1,(200,200)).astype(np.complex64))
  # This will compute the fft to a new GPU array
  d1 = fftn(d0)

  # An in-place transform can also be done by specifying the destination
  d0 = fftn(d0, d0)

  # Or an out-of-place transform to an existing array (the destination array is always returned)
  d1 = fftn(d0, d1)

See the scripts and notebooks in the examples directory.
An example notebook is also `available on google colab
<https://colab.research.google.com/drive/1YJKtIwM3ZwyXnMZfgFVcpbX7H-h02Iej?usp=sharing>`_.
Make sure to select a GPU for the runtime.


Features
--------

- CUDA (using PyCUDA or CuPy) and OpenCL (using PyOpenCL) backends
- C2C, R2C/C2R for inplace and out-of-place transforms
- Direct Cosine Transform (DCT) of type 1, 2, 3 and 4 (EXPERIMENTAL)
- single and double precision for all transforms (double precision requires device support)
- 1D, 2D and 3D transforms.
- array can be have more dimensions than the FFT (batch transforms).
- arbitrary array size, using Bluestein algorithm for prime numbers>13 (note that in this case
  the performance can be significantly lower, up to ~4x, depending on the transform size,
  see example performance plot below)
- transform along a given list of axes - this requires that after collapsing
  non-transformed axes, the last transformed axis is at most along the 3rd dimension,
  e.g. the following axes are allowed: (-2,-3), (-1,-3), (-1,-4), (-4,-5),...
  but not (-2, -4), (-1, -3, -4) or (-2, -3, -4).
  This is not allowed for R2C transforms.
- normalisation=0 (array L2 norm * array size on each transform) and 1 (the backward
  transform divides the L2 norm by the array size, so FFT*iFFT restores the original array)
- unit tests for all transforms: see test sub-directory. Note that these take a **long**
  time to finish due to the exhaustive number of sub-tests.
- Note that out-of-place C2R transform currently destroys the complex array for FFT dimensions >=2
- tested on macOS (10.13.6), Linux (Debian/Ubuntu, x86-64 and power9), and Windows 10
  (Anaconda python 3.8 with Visual Studio 2019 and the CUDA toolkit 11.2)
- inplace transforms do not require an extra buffer or work area (as in cuFFT), unless the x
  size is larger than 8192, or if the y and z FFT size are larger than 2048. In that case
  a buffer of a size equal to the array is necessary. This makes larger FFT transforms possible
  based on memory requirements (even for R2C !) compared to cuFFT. For example you can compute
  the 3D FFT for a 1600**3 complex64 array with 32GB of memory.
- transforms can either be done by creating a VkFFTApp (a.k.a. the fft 'plan'),
  with the selected backend (``pyvkfft.cuda`` for pycuda/cupy or ``pyvkfft.opencl`` for pyopencl)
  or by using the ``pyvkfft.fft`` interface with the ``fftn``, ``ifftn``, ``rfftn`` and ``irfftn``
  functions which automatically detect the type of GPU array and cache the
  corresponding VkFFTApp (see the example notebook pyvkfft-fft.ipynb).

Performance
-----------
See the benchmark notebook, which allows to plot OpenCL and CUDA backend throughput, as well as compare
with cuFFT (using scikit-cuda) and clFFT (using gpyfft).

Example result for batched 2D FFT with array dimensions of batch x N x N using a Titan V:

.. image:: https://raw.githubusercontent.com/vincefn/pyvkfft/master/doc/benchmark-2DFFT-TITAN_V-Linux.png

Notes regarding this plot:

* the computed throughput is *theoretical*, as if each transform axis for the
  couple (FFT, iFFT) required exactly one read and one write. This is obviously not true,
  and explains the drop after N=1024 for cuFFT and (in a smaller extent) vkFFT.
* the batch size is adapted for each N so the transform takes long enough, in practice the
  transformed array is at around 600MB. Transforms on small arrays with small batch sizes
  could produce smaller performances, or better ones when fully cached.
* a number of blue + (CuFFT) are actually performed as radix-N transforms with 7<N<127 (e.g. 11)
  -hence the performance similar to the blue dots- but the list of supported radix transforms
  is undocumented so they are not correctly labeled.

The general results are:

* vkFFT throughput is similar to cuFFT up to N=1024. For N>1024 vkFFT is much more
  efficient than cuFFT due to the smaller number of read and write per FFT axis
  (apart from isolated radix-2 3 sizes)
* the OpenCL and CUDA backends of vkFFT perform similarly, though there are ranges
  where CUDA performs better, due to different cache . [Note that if the card is also used for display,
  then difference can increase, e.g. for nvidia cards opencl performance is more affected
  when being used for display than the cuda backend]
* clFFT (via gpyfft) generally performs much worse than the other transforms, though this was
  tested using nVidia cards. (Note that the clFFT/gpyfft benchmark tries all FFT axis permutations
  to find the fastest combination)

TODO
----

- access to the other backends:

  - for vulkan and rocm this only makes sense combined to a pycuda/cupy/pyopencl equivalent.
- out-of-place C2R transform without modifying the C array ? This would require using a R
  array padded with two wolumns, as for the inplace transform
- half precision ?
- convolution ?
- zero-padding ?
- access to tweaking parameters in VkFFTConfiguration ?
- access to the code of the generated kernels ?
