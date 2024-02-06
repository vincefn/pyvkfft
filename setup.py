# The setup used here is derived with bits from:
# - https://github.com/rmcgibbo/npcuda-example

import os
import sys
import platform
import subprocess
from os.path import join as pjoin
import warnings
from setuptools import setup, find_packages
from setuptools.command.sdist import sdist
from distutils.extension import Extension
from distutils import unixccompiler
from setuptools.command.build_ext import build_ext as build_ext_orig
from setuptools.command.bdist_egg import bdist_egg
from setuptools.command.install_lib import install_lib as su_install_lib
from setuptools.command.sdist import sdist as su_sdist
from pyvkfft.version import __version__, git_version, vkfft_git_version

# Maximum number of dimensions VkFFT can handle. VkFFT sets this to 4,
# pyvkfft uses a default of 8. Set an environment variable
# VKFFT_MAX_FFT_DIMENSIONS to increase this
if 'VKFFT_MAX_FFT_DIMENSIONS' in os.environ:
    VKFFT_MAX_FFT_DIMENSIONS = os.environ['VKFFT_MAX_FFT_DIMENSIONS']
else:
    VKFFT_MAX_FFT_DIMENSIONS = 8


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
    # search for nvcc
    app_name = 'nvcc.exe' if platform.system() == "Windows" else 'nvcc'
    nvcc, home = None, None
    for p in ['CUDA_PATH', 'CUDA_HOME', 'CUDAHOME']:
        if p in os.environ:
            if os.path.exists(pjoin(os.environ[p], 'bin', app_name)):
                nvcc = pjoin(os.environ[p], 'bin', app_name)
                home = os.environ[p]
                break
    if nvcc is None:
        # Search the PATH for NVCC
        nvcc = find_in_path('nvcc.exe', os.environ['PATH'])
        if nvcc is not None:
            home = os.path.dirname(os.path.dirname(nvcc))
        else:
            raise EnvironmentError('The nvcc binary could not be '
                                   'located in your $PATH. Either add it to your path, '
                                   'or set $CUDA_PATH')

    if platform.system() == "Windows":
        libdir = pjoin(home, 'lib', 'x64')
        extra_compile_args = ['-O3', '--ptxas-options=-v', '-Xcompiler', '-MD',
                              f'-DVKFFT_MAX_FFT_DIMENSIONS={VKFFT_MAX_FFT_DIMENSIONS}',
                              "--use-local-env", "-ccbin",
                              os.path.dirname(find_in_path('cl.exe', os.environ['PATH']))]
        tmp = pjoin(os.path.dirname(find_in_path('cl.exe', os.environ['PATH'])).split('bin')[0],
                    'lib', 'x64')
        extra_link_args = ['-L%s' % libdir, "--use-local-env", "-ccbin",
                           os.path.dirname(find_in_path('cl.exe', os.environ['PATH'])),
                           '-L%s' % tmp]
    else:
        if os.path.exists(pjoin(home, 'lib64')):
            libdir = pjoin(home, 'lib64')
        else:
            libdir = pjoin(home, 'lib')
        extra_compile_args = ['-O3', '--ptxas-options=-v', '-std=c++11',
                              '--compiler-options=-fPIC',
                              f'-DVKFFT_MAX_FFT_DIMENSIONS={VKFFT_MAX_FFT_DIMENSIONS}']
        extra_link_args = ['--shared', '-L%s' % libdir]
    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include_dirs': [pjoin(home, 'include'), 'src/VkFFT/vkFFT'],
                  'extra_compile_args': extra_compile_args,
                  'extra_link_args': extra_link_args}
    if 'CONDA_PREFIX' in os.environ:
        cudaconfig['include_dirs'].append(f"{os.environ['CONDA_PREFIX']}/include")
        cudaconfig['include_dirs'].append(f"{os.environ['CONDA_PREFIX']}/Library/include")
    return cudaconfig


def locate_opencl():
    """
    Get the opencl configuration
    :return:
    """
    include_dirs = ['src/VkFFT/vkFFT']
    library_dirs = []
    extra_compile_args = ['-std=c++11', '-Wno-format-security',
                          f'-DVKFFT_MAX_FFT_DIMENSIONS={VKFFT_MAX_FFT_DIMENSIONS}']
    extra_link_args = None
    if platform.system() == 'Darwin':
        libraries = None
        extra_link_args = ['-Wl,-framework,OpenCL']
    elif platform.system() == "Windows":
        # Add include & lib dirs if possible from usual nvidia and AMD paths
        for path in ["CUDA_HOME", "CUDAHOME", "CUDA_PATH"]:
            if path in os.environ:
                include_dirs.append(pjoin(os.environ[path], 'include'))
                library_dirs.append(pjoin(os.environ[path], 'lib', 'x64'))
        library_dirs.append(pjoin(os.path.dirname(find_in_path('cl.exe', os.environ['PATH'])).split('bin')[0],
                                  'lib', 'x64'))
        libraries = ['OpenCL']
        extra_compile_args = [f'/DVKFFT_MAX_FFT_DIMENSIONS={VKFFT_MAX_FFT_DIMENSIONS}']
    else:
        # Linux
        libraries = ['OpenCL']
        extra_link_args = ['--shared']

    if 'CONDA_PREFIX' in os.environ:
        include_dirs.append(f"{os.environ['CONDA_PREFIX']}/include")
        include_dirs.append(f"{os.environ['CONDA_PREFIX']}/Library/include")
        library_dirs.append(f"{os.environ['CONDA_PREFIX']}/Library/Lib")

    opencl_config = {'libraries': libraries, 'extra_link_args': extra_link_args,
                     'include_dirs': include_dirs, 'library_dirs': library_dirs,
                     'extra_compile_args': extra_compile_args}
    return opencl_config


class build_ext_custom(build_ext_orig):
    """Custom `build_ext` command which will correctly compile and link
    the OpenCL and CUDA modules. The hooks are based on the name of the extension"""

    def build_extension(self, ext):
        if "cuda" in ext.name:
            # Use nvcc for compilation. This assumes all sources are .cu files
            # for this extension.
            default_compiler = self.compiler
            # Create unix compiler patched for cu
            self.compiler = unixccompiler.UnixCCompiler()
            self.compiler.src_extensions.append('.cu')
            tmp = CUDA['nvcc']  # .replace('\\\\','toto')
            self.compiler.set_executable('compiler_so', [tmp])
            self.compiler.set_executable('linker_so', [tmp])
            if platform.system() == "Windows":
                CUDA['extra_link_args'] += ['--shared', '-Xcompiler', '-MD']
                # pythonXX.lib must be in the linker paths
                # Is using sys.prefix\libs always correct ?
                CUDA['extra_link_args'].append('-L%s' % pjoin(sys.prefix, 'libs'))

            super().build_extension(ext)
            # Restore default linker and compiler
            self.compiler = default_compiler
        else:
            super().build_extension(ext)

    def get_export_symbols(self, ext):
        """ Hook to make sure we get the correct symbols for windows"""
        if ("opencl" in ext.name or "cuda" in ext.name) and platform.system() == "Windows":
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def get_ext_filename(self, ext_name):
        """ Hook to make sure we keep the correct name (*.so) for windows"""
        if ("opencl" in ext_name or "cuda" in ext_name) and platform.system() == "Windows":
            return ext_name + '.so'
        return super().get_ext_filename(ext_name)


ext_modules = []
install_requires = ['numpy', 'psutil']
exclude_packages = ['examples']
CUDA = None
OPENCL = None

if "VKFFT_BACKEND" in os.environ:
    v = os.environ["VKFFT_BACKEND"]
    # Environment variable to manually select vkfft backends. useful e.g. if nvidia tools
    # are installed but not functional
    # e.g. use:
    #   VKFFT_BACKEND=opencl pip install pyvkfft
    #   VKFFT_BACKEND=cuda pip install .
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
                                   extra_compile_args=CUDA['extra_compile_args'],
                                   include_dirs=CUDA['include_dirs'] + ['src'],
                                   extra_link_args=CUDA['extra_link_args'],
                                   depends=['vkFFT.h']
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
                                 include_dirs=OPENCL['include_dirs'] + ['src'],
                                 libraries=OPENCL['libraries'],
                                 library_dirs=OPENCL['library_dirs'],
                                 extra_link_args=OPENCL['extra_link_args'],
                                 depends=['vkFFT.h']
                                 )

    ext_modules.append(vkfft_opencl_ext)

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()


class bdist_egg_disabled(bdist_egg):
    """ Disabled bdist_egg, to prevent use of 'python setup.py install' """

    def run(self):
        sys.exit("Aborting building of eggs. Please use `pip install .` to install from source.")


# Console scripts, available e.g. as 'pyvkfft-test'
scripts = ['pyvkfft/scripts/pyvkfft_test.py', 'pyvkfft/scripts/pyvkfft_test_suite.py',
           'pyvkfft/scripts/pyvkfft_benchmark.py', 'pyvkfft/scripts/pyvkfft_info.py']

console_scripts = []
for s in scripts:
    s1 = os.path.splitext(os.path.split(s)[1])[0]
    s0 = os.path.splitext(s)[0]
    console_scripts.append("%s = %s:main" % (s1.replace('_', '-'), s0.replace('/', '.')))


class pyvkfft_sdist(su_sdist):
    """Hook to include git version of pyvkfft and VkFFT"""

    def make_release_tree(self, base_dir, files):
        super(pyvkfft_sdist, self).make_release_tree(base_dir, files)
        try:
            # Replace git_version_placeholder by real git version
            version_file = os.path.join(base_dir, "pyvkfft/version.py")
            vers = open(version_file).read()
            os.remove(version_file)
            with open(version_file, "w") as fh:
                vers = vers.replace("vkfft_git_version_placeholder", vkfft_git_version())
                vers = vers.replace("git_version_placeholder", git_version())
                fh.write(vers)
        except:
            print("sdist: replacing git_version failed")


class pyvkfft_install_lib(su_install_lib):
    """Hook to include git version of pyvkfft and VkFFT"""

    def run(self):
        super(pyvkfft_install_lib, self).run()
        try:
            # print(self.install_dir, self.build_dir)
            # Replace git_version_placeholder by real git version
            version_file = os.path.join(self.install_dir, "pyvkfft/version.py")
            vers = open(version_file).read()
            os.remove(version_file)
            with open(version_file, "w") as fh:
                vers = vers.replace("vkfft_git_version_placeholder", vkfft_git_version())
                vers = vers.replace("git_version_placeholder", git_version())
                fh.write(vers)
        except:
            print("install_lib: replacing git_version failed")


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
          "Source": "https://github.com/vincefn/pyvkfft",
          "Documentation": "https://pyvkfft.readthedocs.io/",
          "Bug Tracker": "https://github.com/vincefn/pyvkfft/issues",
          "VkFFT project": "https://github.com/DTolm/VkFFT",
      },
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Environment :: GPU",
      ],
      license='MIT License',
      cmdclass={'build_ext': build_ext_custom,
                'bdist_egg': bdist_egg if 'bdist_egg' in sys.argv else bdist_egg_disabled,
                'sdist': pyvkfft_sdist,
                'install_lib': pyvkfft_install_lib
                },
      install_requires=install_requires,
      extras_require={'doc': ['sphinx', 'nbsphinx', 'nbsphinx-link', 'sphinx-argparse',
                              'sphinx-rtd-theme']},
      test_suite="test",
      entry_points={'console_scripts': console_scripts},
      )
