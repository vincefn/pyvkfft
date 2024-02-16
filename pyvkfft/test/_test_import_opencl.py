# -*- coding: utf-8 -*-

# PyVkFFT
#   (c) 2024- : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr
#
#
# Basic test imports of the opencl library
# Primarily used to check build status.

from ..base import load_library

_vkfft_opencl = load_library("_vkfft_opencl")
