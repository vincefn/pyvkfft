# -*- coding: utf-8 -*-

# PyVkFFT
#   (c) 2024- : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr
#
#
# Basic test imports of the opencl library
# Primarily used to check build status.

import os
from ..base import _library_path, load_library

assert os.path.exists(_library_path("_vkfft_opencl")), "_vkfft_opencl shared library not found"
# _vkfft_opencl = load_library("_vkfft_opencl")
