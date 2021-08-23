# Most of the setup used here is derived from https://github.com/rmcgibbo/npcuda-example
# License: MIT

import os
import sys
from os.path import join as pjoin
import warnings
from setuptools import setup, find_packages
from setuptools.command.sdist import sdist, sdist_add_defaults
from distutils.extension import Extension
from Cython.Distutils import build_ext
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

    Starts by looking for the CUDAHOME env variable. If not found,
    everything is based on finding 'nvcc' in the PATH.
    """

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
                                   'or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))
    if os.path.exists(pjoin(home, 'lib64')):
        libdir = 'lib64'
    else:
        libdir = 'lib'
    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, libdir)}
    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be '
                                   'located in %s' % (k, v))

    return cudaconfig


def locate_opencl():
    """
    Get the opencl configuration
    :return:
    """
    if 'darwin' in sys.platform:
        libraries = None
        extra_link_args = ['-Wl,-framework,OpenCL']  # , '--shared'
    else:
        libraries = ['OpenCL']
        extra_link_args = ['--shared']

    return {'libraries': libraries, 'extra_link_args': extra_link_args}


def customize_compiler_for_nvcc(self):
    """Inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """

    # Tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # Save references to the default compiler_so and _compile methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # Now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            self.set_executable('linker_so', CUDA['nvcc'])

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # Reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # Inject our redefined _compile method into the class
    self._compile = _compile


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


# Run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


ext_modules = []
install_requires = ['numpy']
exclude_packages = ['examples', 'test']
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
                                                       '--compiler-options=-fPIC']
                                   ,
                                   include_dirs=[CUDA['include'], 'src'],
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
        warnings.warn("CUDA not available ($CUDAHOME variable missing and nvcc not in path. "
                      "Skipping pyvkfft.cuda module installation.", UserWarning)


if 'opencl' not in exclude_packages:
    OPENCL = locate_opencl()
    install_requires.append('pyopencl')

    # OpenCL extension
    vkfft_opencl_ext = Extension('pyvkfft._vkfft_opencl',
                                 sources=['src/vkfft_opencl.cpp'],
                                 extra_compile_args=['-std=c++11', '-Wno-format-security'],
                                 libraries=OPENCL['libraries'],
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

      cmdclass={'build_ext': custom_build_ext, 'sdist_vkfft': sdist_vkfft},
      install_requires=install_requires,
      test_suite="test")
