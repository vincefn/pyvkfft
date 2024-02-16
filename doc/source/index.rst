.. pyvkfft documentation master file, created by
   sphinx-quickstart on Wed Aug  2 20:07:20 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyvkfft
=======

.. include:: ../../README.rst
   :end-before: Installation

See the:

* :doc:`List of features <features>`
* :doc:`Performance details <performance>`
* :doc:`Accuracy tests <accuracy>`


Installation
============
Requirements: ``pyvkfft`` only requires ``numpy``, ``psutil``, and at least one
GPU computing package among ``pyopencl``, ``pycuda`` and ``cupy``.

You can install ``pyvkfft`` either using PyPI:

.. code-block:: shell

   pip install pyvkfft

or conda (using the conda-forge channel):

.. code-block:: shell

   conda config --add channels conda-forge
   conda install pyvkfft

When using cuda, you can also specify the nvrtc library version
using the ``cuda-version`` package:

.. code-block:: shell

   # Example also with pyopencl and cupy
   conda install pyvkfft pyopencl cupy cuda-version=12

Driver/toolkit requirements
---------------------------
``pyvkfft`` needs a working GPU computing environment, including drivers,
compiling tools and libraries.

For **OpenCL**, the Installable Client Driver (ICD) should be available - it is normally
provided automatically when installing GPU toolkits (cuda, orcm,...). When using conda,
it may be necessary to install ``ocl-icd-system`` (under Linux) or ``ocl-icd-wrapper-apple``.
This is done automatically if ``pyvkfft`` is installed using conda.

For **CUDA**, in addition to the driver, the cuda toolkit (including ``nvcc`` and the
real-time-compute (rtc) libraries) must also be installed. The compiler and toolkit
can be installed using ``conda`` if they are not available system-wide.

When installing using pip (needs compilation), the path to ``nvcc``
(or ``nvcc.exe``) will be automatically searched, first using the ``CUDA_PATH``
or ``CUDA_HOME`` environment variables, or then in the ``PATH``.
If ``nvcc`` is not found, only support for OpenCL will be compiled.

Windows installation (cuda)
---------------------------
Windows installation can be tricky. Here are some hints for windows 10 with an
nvidia card - YMMV:

* first install Visual studio (tested with 2019) with C++ tools and windows SDK
* install conda using `mambaforge <https://github.com/conda-forge/miniforge/releases>`_
  which has the advantage of including ``conda-forge`` as a default channel, and
  ``mamba`` for faster installations

Then you can either install all packages using conda:

.. code-block:: shell

   mamba create -n myenv python=3.11 numpy scipy matplotlib ipython psutil pyopencl cupy pycuda nvidia::cuda pyvkfft

or install all packages except pyvkfft using conda:

.. code-block:: shell

   mamba create -n myenv python=3.11 numpy scipy matplotlib ipython psutil pyopencl cupy pycuda nvidia::cuda
   conda activate myenv
   pip install pyvkfft

or install using conda for base packages including nvidia::cuda, then pypi for
pyopencl, pycuda, cupy and pyvkfft

.. code-block:: shell

   mamba create -n myenv python=3.11 numpy scipy matplotlib ipython psutil pyopencl cupy pycuda nvidia::cuda
   conda activate myenv
   pip install pyopencl pycuda cupy-cuda12x pyvkfft

Troubleshooting installation
----------------------------
If you encounter issues, make sure you have the right combination of toolkit
and driver. Also, note that installing using ``pip`` is cached, so if you change
your configuration (new toolkit version), you must make sure to recompile the
``pycuda`` and ``pyvkfft`` packages, using e.g.:

``pip install pycuda pyvkfft --no-cache``

The ``pyvkfft-info`` script can give you some information about current support
for OpenCL/CUDA, the version of the driver and the toolkit, and the detected
GPU devices.

Example
=======

.. include:: ../../README.rst
   :start-after: Examples
   :end-before: See the scripts

:doc:`Notebook examples <examples/index>`
=========================================

:doc:`API <api/index>`
====================================

:doc:`Command-line scripts <scripts/index>`
===========================================

:doc:`changelog`
================

Authors & acknowledgements
==========================

.. include:: ../../README.rst
   :start-after: acknowledgements


Indices and tables
==================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   accuracy
   changelog
   features
   performance
   examples/index
   api/index
   scripts/index

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

