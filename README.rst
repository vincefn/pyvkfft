pyvkfft - python interface to the CUDA backend of VkFFT (Vulkan Fast Fourier Transform library)
===============================================================================================

`VkFFT <https://github.com/DTolm/VkFFT>`_ is a GPU-accelerated Fast Fourier Transform library
for Vulkan/CUDA/HIP projects.

pyvkfft offers a basic python interface to the **CUDA backend of VkFFT**, compatible with pyCUDA.

*The code should now be in a working state, and passes all unit tests, and no errors are reported by either valgrind or cuda-memcheck.*

Installation
------------

Requirements:

- `vkfft.h` installed in the usual include directories
- `pycuda` and CUDA developments tools (`nvcc`)
- `numpy`

This package should be installed using pip or `python setup.py install`.

Examples
--------
See the script and notebook in the examples directory.
The notebook is also `available on google colab
<https://colab.research.google.com/drive/1YJKtIwM3ZwyXnMZfgFVcpbX7H-h02Iej?usp=sharing>`_.
Make sure to select a GPU for the runtime.


Status
------
What works:

- C2C, R2C/C2R for inplace and out-of-place transforms
- single and double precision for all transforms
- all transforms accept 1D, 2D and 3D arrays, with the FT dimension <= array dimension
- normalisation=0 (array L2 norm * array size on each transform) and 1 (the backward
  transform divides the L2 norm by the array size, so FFT*iFFT restores the original array)
- now testing the FFT size does not exceed the allowed maximum prime number decomposition (13)
- unit tests for all transforms: use `python setup.py test`

TODO
----

- access to the other backends: OpenCL. As for vulkan and rocm this only makes sense combined to a pycuda/cupy/pyopencl equivalent. 
- half precision
- convolution ?
- access to tweaking parameters in VkFFTConfiguration ?
- access to the code of the generated kernels ?
