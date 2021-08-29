# The setup used here is derived with bits from:
# - https://github.com/rmcgibbo/npcuda-example

import os
import platform
from os.path import join as pjoin
import warnings
from setuptools import setup, find_packages
from setuptools.command.sdist import sdist, sdist_add_defaults
from distutils.extension import Extension
from setuptools.command.build_ext import build_ext as build_ext_orig
from pyvkfft.version import __version__


def find_in_path(name, path):
    """Find a file in a search path"""

    # Adapted fom http://code.activestate.com/recipes/52224
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME or CUDA_PATH env variable.
    If not found, find 'nvcc' in the PATH.
    """
    if platform.system() == "Windows":
        if 'CUDA_PATH' in os.environ:
            home = os.environ['CUDA_PATH']
            nvcc = pjoin(home, 'bin', 'nvcc.exe')
        else:
            # Otherwise, search the PATH for NVCC
            nvcc = find_in_path('nvcc.exe', os.environ['PATH'])
            if nvcc is None:
                raise EnvironmentError('The nvcc binary could not be '
                                       'located in your $PATH. Either add it to your path, '
                                       'or set $CUDA_PATH')
            home = os.path.dirname(os.path.dirname(nvcc))
        libdir = pjoin(home, 'lib', 'x64')
        print("locate_cuda: Windows -> ", nvcc)
    else:
        # First check if the CUDAHOME env variable is in use
        if 'CUDAHOME' in os.environ:
            home = os.environ['CUDAHOME']
            nvcc = pjoin(home, 'bin', 'nvcc')
        else:
            # Otherwise, search the PATH for NVCC
            nvcc = find_in_path('nvcc', os.environ['PATH'])
            if nvcc is None:
                raise EnvironmentError('The nvcc binary could not be '
                                       'located in your $PATH. Either add it to your path, '
                                       'or set $CUDAHOME or $CUDA_PATH')
            home = os.path.dirname(os.path.dirname(nvcc))
        if os.path.exists(pjoin(home, 'lib64')):
            libdir = pjoin(home, 'lib64')
        else:
            libdir = pjoin(home, 'lib')
    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': libdir}
    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be '
                                   'located in %s' % (k, v))
    print("CUDA config: ", cudaconfig)
    return cudaconfig


def locate_opencl():
    """
    Get the opencl configuration
    :return:
    """
    include_dirs = []
    library_dirs = []
    extra_compile_args = ['-std=c++1L', '-Wno-format-security']
    extra_link_args = None
    if platform.system() == 'Darwin':
        libraries = None
        extra_link_args = ['-Wl,-framework,OpenCL', '--shared']
    elif platform.system() == "Windows":
        # Add include & lib dirs if possible from usual nvidia and AMD paths
        for path in ["CUDA_HOME", "CUDAHOME", "CUDA_PATH"]:
            if path in os.environ:
                include_dirs.append(pjoin(os.environ[path], 'include'))
                library_dirs.append(pjoin(os.environ[path], 'lib', 'x64'))
        libraries = ['OpenCL']
        extra_compile_args = None
    else:
        # Linux
        libraries = ['OpenCL']
        extra_link_args = ['--shared']

    opencl_config = {'libraries': libraries, 'extra_link_args': extra_link_args,
                    'include_dirs': include_dirs, 'library_dirs': library_dirs,
                    'extra_compile_args': extra_compile_args}
    print("OpenCL config: ", opencl_config)
    return opencl_config


class build_ext_custom(build_ext_orig):
    """Custom `build_ext` command which will correctly compile and link
    the OpenCL and CUDA modules."""

    def get_export_symbols(self, ext):
        """ Hook based on the name to make sure we get the correct symbols"""
        if "opencl" in ext.name:
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def get_ext_filename(self, ext_name):
        """ Hook based on the name to make sure we keep the correct name ('.so)"""
        if "opencl" in ext_name:
            return ext_name + '.so'
        return super().get_ext_filename(ext_name)


class sdist_vkfft(sdist):
    """
    Sdist overloaded to get vkfft header, readme and license from VkFFT's git
    """

    def run(self):
        # Get the latest vkFFT.h from github
        os.system('curl -L https://raw.githubusercontent.com/DTolm/VkFFT/master/vkFFT/vkFFT.h -o src/vkFFT.h')
        os.system('curl -L https://raw.githubusercontent.com/DTolm/VkFFT/master/LICENSE -o LICENSE_VkFFT')
        os.system('curl -L https://raw.githubusercontent.com/DTolm/VkFFT/master/README.md -o README_VkFFT.md')
        super(sdist_vkfft, self).run()


ext_modules = []
install_requires = ['numpy']
exclude_packages = ['examples', 'test', 'cuda']
CUDA = None
OPENCL = None

for k, v in os.environ.items():
    if "VKFFT_BACKEND" in k:
        # Kludge to manually select vkfft backends. useful e.g. if nvidia tools
        # are installed but not functional
        # e.g. use:
        #   VKFFT_BACKEND=cuda,opencl python setup.py install
        #   VKFFT_BACKEND=opencl pip install pyvkfft
        if 'opencl' not in v.lower():
            exclude_packages.append('opencl')
        if 'cuda' not in v.lower():
            exclude_packages.append('cuda')

if 'cuda' not in exclude_packages:
    try:
        CUDA = locate_cuda()
        vkfft_cuda_ext = Extension('pyvkfft._vkfft_cuda',
                                   sources=['src/vkfft_cuda.cu'],
                                   libraries=['nvrtc', 'cuda'],
                                   # This syntax is specific to this build system
                                   # we're only going to use certain compiler args with nvcc
                                   # and not with gcc the implementation of this trick is in
                                   # customize_compiler()
                                   extra_compile_args=['-O3', '--ptxas-options=-v', '-std=c++11',
                                                       '--compiler-options=-fPIC'],
                                   include_dirs=[CUDA['include']],
                                   extra_link_args=['--shared', '-L%s' % CUDA['lib64']]
                                   )
        ext_modules.append(vkfft_cuda_ext)
        # install_requires.append("pycuda")
        try:
            import pycuda

            has_pycuda = True
        except ImportError:
            has_pycuda = False
        try:
            import cupy
        except ImportError:
            if has_pycuda is False:
                print("Reminder: you need to install either PyCUDA or CuPy to use pyvkfft.cuda")
    except:
        exclude_packages.append('cuda')
        warnings.warn("CUDA not available ($CUDAHOME/$CUDA_PATH variables missing "
                      "and nvcc not in path. "
                      "Skipping pyvkfft.cuda module installation.", UserWarning)

if 'opencl' not in exclude_packages:
    OPENCL = locate_opencl()
    install_requires.append('pyopencl')

    # OpenCL extension
    vkfft_opencl_ext = Extension('pyvkfft._vkfft_opencl',
                                 sources=['src/vkfft_opencl.cpp'],
                                 extra_compile_args=OPENCL['extra_compile_args'],
                                 include_dirs=OPENCL['include_dirs'],
                                 libraries=OPENCL['libraries'],
                                 library_dirs=OPENCL['library_dirs'],
                                 extra_link_args=OPENCL['extra_link_args']
                                 )

    ext_modules.append(vkfft_opencl_ext)

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name="pyvkfft",
      version=__version__,
      description="Python wrapper for the CUDA and OpenCL backends of VkFFT,"
                  "providing GPU FFT for PyCUDA, PyOpenCL and CuPy",
      long_description=long_description,
      ext_modules=ext_modules,
      packages=find_packages(exclude=exclude_packages),
      include_package_data=True,
      author="Vincent Favre-Nicolin",
      author_email="favre@esrf.fr",
      url="https://github.com/vincefn/pyvkfft",
      project_urls={
          "Bug Tracker": "https://github.com/vincefn/pyvkfft/issues",
          "VkFFT project": "https://github.com/DTolm/VkFFT",
      },
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
          "Operating System :: OS Independent",
          "Environment :: GPU",
      ],

      cmdclass={'build_ext': build_ext_custom, 'sdist_vkfft': sdist_vkfft},
      install_requires=install_requires,
      test_suite="test")
