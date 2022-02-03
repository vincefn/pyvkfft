Version 2022.1 (2022-02-03)
-----------------------------
* Added accuracy unit tests, which can be used systematically
  using the 'pyvkfft-test' installed script
* An extensive testing is now made before official releases,
  evaluating all type of transforms (c2c, r2c, dct, 1, 2 and 3D,
  in and out-of-place, norm 0 and 1), different GPUs, both OpenCL
  and CUDA, etc... Comparison is made against pyfftw, scipy or numpy.
* Update to VkFFT 1.2.21, with support for DCT types 1, 2, 3 and 4,
  also fixing a number of issues (see closed issues at
  https://github.com/DTolm/VkFFT/issues), and passing all tests
  on different GPUs (OpenCL and CUDA, AMD and nVidia)
* Raise a RuntimeError if the VkFFTApp initialisation or the
  GPU kernel launch fails, with the corresponding VkFFT error.
* [BUG] Correct inverse FFT calculation using pyvkfft.fft.ifftn()
* Installation from source using 'python setup.py install' is now
  disabled - 'pip install' should always be used.
* Added config.USE_LUT and config.FFT_CACHE_NB variables, which
  can be used to modify the default behaviour, and can also be set
  e.g. with the PYVKFFT_USE_LUT environment variable.

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
