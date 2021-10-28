Version 2021.2.2 (2021-10-XX)
-----------------------------
* Update to VkFFT 1.2.X, with support for DCT types 1, 2, 3 and 4,
  also fixing DCT issues (see https://github.com/DTolm/VkFFT/issues/48).
  DCT calculations now match scipy.
* [BUG] Correct inverse FFT calculation using pyvkfft.fft.ifftn()

Version 2021.2.1 (2021-09-04)
-----------------------------
* Support for windows installation (cuda and opencl) [requires visual studio
  with c++ tools and the cuda toolkit with nvcc. Untested with the AMD SDK]
* Remove Cython reference in setup.py

Version 2021.2 (2021-08-23)
---------------------------
* Enable transforms for any array size (VkFFT Bluestein algorithm)
* Allow 3D transforms on arrays with >3 dimensions (batch transform)
* Support for transforms on a given list of axes, instead of
  only the first ndim axes. Unavailable for R2C.
* Added a simple pyvkfft.fft interface with `fftn`, `ifftn`, `rfftn`, `irfftn`
  functions which automatically recognize the type of GPU arrays
  and cache the generated VkFFTApp (FFT plans).
* Direct Cosine Transform (DCT) of type 2, 3 and 4 (EXPERIMENTAL)
* Support CuPy arrays in addition to PyCUDA and PyOpenCL
* OpenCL: test for half and double-precision support
* OpenCL: relax accuracy requirements in unit tests
* Fix shape test for out-of-place R2C transforms
* Add a base VkFFTApp class common to OpenCL and CUDA
* Installation: fix macOS compilation. Allow selection of backends
  from an environment variable

Version 2021.1b6 (2021-05-02)
-----------------------------
* Initial release, in phase with VkFFT 1.2.2
