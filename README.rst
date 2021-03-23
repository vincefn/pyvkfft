pyvkfft - python interface to VkFFT (Vulkan Fast Fourier Transform library)
===========================================================================

`VkFFT <https://github.com/DTolm/VkFFT>`_ is a GPU-accelerated Fast Fourier Transform library
for Vulkan/CUDA/HIP projects.

pyvkfft offers a basic python interface to the CUDA backend of VkFFT, compatible with pyCUDA.

*This is very preliminary, mostly a proof-of concept, and may be unrelaible or prone to
errors or memory leaks. Use at your own risks.*

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

- C2C inplace transforms, single and double precision
- C2C out-of-place transform, single and double precision (now keeps the original data)
- normalisation=0 (array L2 norm * array size on each transform) and 1 (the backward
  transform divides the L2 norm by the array size, so FFT*iFFT restores the original array)
- R2C inplace, single and double precision
- now testing the FFT size does not exceed the allowed maximum prime number decomposition (13)
- small testsuite: use `python setup.py test`

TODO
----

- access to the other backends (vulkan, rocm) ? Not useful unless combined to a pycuda equivalent.
- half precision
- convolution ?
- access to tweaking parameters in VkFFTConfiguration ?
- access to the code of the generated kernels ?
