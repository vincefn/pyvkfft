pyvkfft - python interface to VkFFT (Vulkan Fast Fourier Transform library)
===========================================================================

`VkFFT <https://github.com/DTolm/VkFFT` is a GPU-accelerated Fast Fourier Transform library
for Vulkan/CUDA/HIP projects.

pyvkfft offers a basic python interface to the CUDA backend of VkFFT, compatible with pyCUDA.

*This is very preliminary, mostly a proof-of concept, and may be unrelaible or prone to
errors or memory leaks. Use at your own risks.*

Installation
------------

Requirements:

- `vkfft.h` installed in the usual include directories, or in `pyvkfft/src/`
- `pycuda` and CUDA developments tools (`nvcc`)
- `numpy`

This package should be installed using pip or `python setup.py install`.

TODO
----

- access to the other backends
- check using multiple plans/VkFFT applications
- check chaining other calculations and FFT
- ...

