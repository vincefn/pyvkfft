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

Example notebook
----------------
You can try a `notebook on google colab
<https://colab.research.google.com/drive/1YJKtIwM3ZwyXnMZfgFVcpbX7H-h02Iej?usp=sharing>`_.
Make sure to select a GPU for the runtime.

TODO
----

- access to the other backends
- check using multiple plans/VkFFT applications
- check chaining other calculations and FFT
- support for:

  - half precision
  - double precision
  - out-of-place transforms
  - normalisation of inverse transform
  - real<->complex transforms
  - convolution
  - using multiple streams
  - access to tweaking parameters in VkFFTConfiguration
