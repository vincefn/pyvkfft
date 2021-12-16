# -*- coding: utf-8 -*-

__authors__ = ["Vincent Favre-Nicolin (pyvkfft), Dmitrii Tolmachev (VkFFT)"]
__license__ = "MIT"
__date__ = "2021/09/04"
# Valid numbering includes 3.1, 3.1.0, 3.1.2, 3.1dev0, 3.1a0, 3.1b0
__version__ = "2021.2.1"


def vkfft_version():
    """
    Get VkFFT version
    :return: version as X.Y.Z
    """
    # We import here as otherwise it would mess with setup.py which reads __version__
    # while the opencl library has not yet been compiled.
    try:
        from .opencl import vkfft_version as cl_vkfft_version
        return cl_vkfft_version()
    except ImportError:
        # On some platforms (e.g. pp64le) opencl may not be available while cuda is
        try:
            from .cuda import vkfft_version as cu_vkfft_version
            return cu_vkfft_version()
        except ImportError:
            raise ImportError("Neither cuda or opencl vkfft_version could be imported")
