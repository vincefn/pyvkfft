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

You can install ``pyvkfft`` either using PyPI:

.. code-block:: shell

   pip install pyvkfft

or conda (using the conda-forge channel):

.. code-block:: shell

   conda config --add channels conda-forge
   conda install pyvkfft

Example
=======

.. include:: ../../README.rst
   :start-after: Examples
   :end-before: See the scripts

See :doc:`other notebook examples <examples/index>`

API Documentation
=================
* :doc:`api/simple-fft`
* :doc:`api/core-fft`
* :doc:`api/testing`

Command-line scripts
====================
* :doc:`scripts/benchmark`
* :doc:`scripts/test`
* :doc:`scripts/test-suite`


:doc:`changelog`
================


Indices and tables
==================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   accuracy
   changelog
   features
   performance
   api/core-fft
   api/simple-fft
   api/testing
   scripts/benchmark
   scripts/test
   scripts/test-suite

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

