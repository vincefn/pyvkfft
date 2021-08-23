# -*- coding: utf-8 -*-

__authors__ = ["Vincent Favre-Nicolin (pyvkfft), Dmitrii Tolmachev (VkFFT)"]
__license__ = "MIT"
__date__ = "2021/08/23"
# Valid numbering includes 3.1, 3.1.0, 3.1.2, 3.1dev0, 3.1a0, 3.1b0
__version__ = "2021.2"


def vkfft_version():
    """
    Get VkFFT version
    :return: version as X.Y.Z
    """
    # We import here as otherwise it would mess with setup.py which reads __vresion__
    # while the opencl library as not yet been compiled.
    from .opencl import vkfft_version as cl_vkfft_version
    return cl_vkfft_version()
