Version 2021.X (2021-XX-XX)
---------------------------
* Enable transforms for any array size (VkFFT Bluestein algorithm)
* Allow 3D transforms on arrays with >3 dimensions (batch transform)
* Support for transforms on a given list of axes, instead of
  only the first ndim axes. Unavailable for R2C.
* Direct Cosine Transform (DCT) of type 2, 3 and 4.
* OpenCL: test for half and double-precision support
* OpenCL: relax accuracy requirements in unit tests
* Fix shape test for out-of-place R2C transforms
* Add a base VkFFTApp class common to OpenCL and CUDA
* Installation: fix macOS compilation. Allow selection of backends
  from an environment variable

Version 2021.1b6 (2021-05-02)
-----------------------------
* Initial release, in phase with VkFFT 1.2.2
