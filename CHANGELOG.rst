Version 2024.1.2 (2024-02-17)
-----------------------------
* Fix conda installation with specified ``cuda-version``,
  notably for cuda 12.x support
* add conda-forge build test for cuda and opencl libraries

Version 2024.1.1 (2024-02-12)
-----------------------------
* Fix pycuda initialisation during accuracy tests (pyvkff-test).

Version 2024.1 (2024-02-06)
-----------------------------
* Based on VkFFT 1.3.4
* Add support for direct sine transforms (DST)
* R2C, DST and DCT now support arbitrary sizes (up to ~2^32,
  same as C2C)
* Odd lengths for the fast axis is now supported for all R2C
  transforms. Inplace transforms require using
  the r2c_odd=True parameter
* Custom transform axes (strided) are now allowed also for R2C,
  as long as the fast axis is transformed.
* added functions to access the size of the temporary buffer
  created by VkFFT (if any), the type of algorithm used along
  each axis (radix, Rader, Bluestein), and the number of
  uploads for each transformed axis.
* DCT and DST now support F-ordered arrays
* Longer default test including multi-upload using radix,
  Rader and Bluestein algorithms.
* The full test suite (including c2c, r2c, dct, dst, radix
  and non-radix transforms, single and double precision)
  now includes about 1.5 million unit tests
* The pyvkff-benchmark script can also test R2C, DCT and DST
  transforms, and will give more details about the algorithm
  used for performance tuning.
* Added pyvkfft-info script

Version 2023.2.post1 (2023-09-21)
-----------------------------
* Include doc in manifest

Version 2023.2 (2023-08-14)
-----------------------------
* Based on VkFFT 1.3.1
* Add support for more than 3 FFT dimensions (defaults to 8
  in pyvkfft, can be changed when installing)
* Add options to manually or automatically tune the FFT performance
  for the used GPUs.
* Add pyvkfft-benchmark script.
* The VkFFT source is now included as a git subproject
* Actually use cuda_stream parameter in the pyvkfft.fft interface
* Take into account current cuda device when automatically
  caching VkFFTApp using the pyvkfft.fft interface(#26)
  This enable using multiple GPUs in a multi-threaded approach.
* The pyvkfft-test will ignore the PoCL OpenCL platform if
  another is available and unless it is explicitly selected.

Version 2023.1.1 (2023-01-22)
-----------------------------
* Fix MANIFEST.in including vkfft_cuda.cu for python setup.py sdist

Version 2023.1 (2023-01-19)
-----------------------------
* VkFFT 1.2.33, now using Rader algorithm for better performance
  with many non-radix sizes.
* Fix R2C tests when using numpy (scipy unavailable) [#19]
* Add support for F-ordered arrays (C2C and R2C)
* Allow selection of backend for non-systematic pvkfft-test
* Add parameter to select the OpenCL platform in pyvkfft-test
* For pyopencl, transforms will use the array's queue by default
  instead of the application's (a warning will be written to
  notify this change when they differ; it can be disabled using
  config.WARN_OPENCL_QUEUE_MISMATCH). A queue can also be supplied
  to the fft() and ifft() methods.
  (from @isuruf, https://github.com/vincefn/pyvkfft/pull/17)
* Fix simple fft interface import when only pycuda is used
* Add cuda_driver_version, cuda_compile_version, cuda_runtime_version
  functions.
* Add simpler interface to run benchmarks, using separate processes.
* add pyvkfft-test-suite for long tests (up to 30 hours) for validation
  before new releases.

Version 2022.1.1 (2022-02-14)
-----------------------------
* Correct the dtype of the returned array for fft.rfftn() and fft.irfftn()
  in the case of an inplace transform
* Pycuda: cast the gpudata pointer to int for comparisons
* Fix TestFFT colour attribute default value

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
