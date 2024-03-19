# -*- coding: utf-8 -*-

__authors__ = ["Vincent Favre-Nicolin (pyvkfft), Dmitrii Tolmachev (VkFFT)"]
__license__ = "MIT"
__date__ = "2024/03/19"
# Valid numbering includes 3.1, 3.1.0, 3.1.2, 3.1dev0, 3.1a0, 3.1b0, 3.1.2.post1,...
__version__ = "2024.1.2.post0"

import os


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


def git_version():
    """
    Get the full pyvkfft version name with git hash, e.g. "2020.1-65-g958b7254-dirty"
    Only works if the current directory is part of the git repository, or
    after installation when the placeholder string has been replaced.
    :return: the pyvkfft git version, or "unknown"
    """
    # in distributed & installed versions this is replaced by a string
    __git_version_static__ = "git_version_placeholder"
    if "placeholder" not in __git_version_static__:
        return __git_version_static__
    from subprocess import Popen, PIPE
    try:
        p = Popen(['git', 'describe', '--tags', '--dirty', '--always'],
                  stdout=PIPE, stderr=PIPE)
        return p.stdout.readlines()[0].strip().decode("UTF-8")
    except:
        return "unknown"


def vkfft_git_version():
    """
    Get the full vkfft version name with git hash, e.g. "2020.1-65-g958b7254-dirty"
    Only works if the current directory is part of the git repository, or
    after installation when the placeholder string has been replaced.
    :return: the vkfft git version, or "unknown"
    """
    from subprocess import Popen, PIPE
    # in distributed & installed versions this is replaced by a string
    __git_version_static__ = "vkfft_git_version_placeholder"
    if "placeholder" not in __git_version_static__:
        return __git_version_static__
    try:
        p = Popen(['git', 'describe', '--tags', '--dirty', '--always'],
                  cwd=os.path.join(os.getcwd(), 'src', 'VkFFT'), stdout=PIPE, stderr=PIPE)
        return p.stdout.readlines()[0].strip().decode("UTF-8")
    except:
        return "unknown"
