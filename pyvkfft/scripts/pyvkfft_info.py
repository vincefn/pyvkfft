# -*- coding: utf-8 -*-

# PyVkFFT
#   (c) 2024- : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr
#
#
# pyvkfft script with info about VkFFT version and CUDA/OpenCL support


from pyvkfft.version import git_version, vkfft_git_version, __version__

vkfft_ver = None
try:
    from pyvkfft.cuda import has_pycuda, has_cupy, vkfft_version as cu_vkfft_version, \
        cuda_compile_version, cuda_runtime_version, cuda_driver_version

    vkfft_ver = cu_vkfft_version()
except (ImportError, OSError):
    has_cupy, has_pycuda = False, False

try:
    import pyopencl as cl

    has_opencl = True
    from pyvkfft.opencl import vkfft_version as cl_vkfft_version

    if vkfft_ver is None:
        vkfft_ver = cl_vkfft_version()

except (ImportError, OSError):
    has_opencl = False


def main():
    print(f"pyvkfft version: {__version__:12s} [git: {git_version()}]")
    print(f"VkFFT version:   {vkfft_ver:12s} [git: {vkfft_git_version()}]")
    print(f"\nCUDA support: {has_pycuda or has_cupy}")
    if has_cupy or has_pycuda:
        print(f"  CUDA driver version:  {cuda_driver_version()}")
        print(f"  CUDA runtime version: {cuda_runtime_version()}")
        print(f"  CUDA compile version: {cuda_compile_version()}")

        if has_pycuda:
            from pycuda import VERSION_TEXT
            print(f"  pycuda available: {has_pycuda} , version={VERSION_TEXT}")
        else:
            print(f"  pycuda available: {has_pycuda}")
        if has_cupy:
            from cupy._version import __version__ as cp_ver

            print(f"  cupy available:   {has_cupy} , version={cp_ver}")
        else:
            print(f"  cupy available:   {has_cupy}")
        # Devices
        if has_pycuda:
            import pycuda.autoprimaryctx
            import pycuda.driver as cu_drv
            print(f"  #CUDA devices:   {cu_drv.Device.count()} (pycuda)")
            for i in range(cu_drv.Device.count()):
                print(f"       {i}: {cu_drv.Device(i).name()}")
        elif has_cupy:
            import cupy as cp
            print(f"  #CUDA devices:   {cp.cuda.runtime.getDeviceCount()} (cupy)")
            for i in range(cp.cuda.runtime.getDeviceCount()):
                print(f"       {i}: {cp.cuda.runtime.getDeviceProperties(i)['name'].decode()}")

    print(f"\nOpenCL support: {has_opencl}")
    if has_opencl:
        from pyopencl.version import VERSION_TEXT
        print(f"PyOpenCL version: {VERSION_TEXT}")
        print(f"  OpenCL platform and devices (GPU only):")
        for p in cl.get_platforms():
            nb_gpu = sum([d.type & cl.device_type.GPU for d in p.get_devices()])
            print(f"    platform: {p.name}")
            print(f"      Vendor:  {p.get_info(cl.platform_info.VENDOR)}")
            print(f"      Version: {p.get_info(cl.platform_info.VERSION)}")
            print(f"      #GPU devices: {nb_gpu}:")
            for d in p.get_devices():
                if d.type & cl.device_type.GPU:
                    print(f"        {d.name}")
                    try:
                        # CL_DEVICE_BOARD_NAME_AMD                        0x4038
                        print(f"          Board Name (AMD): {d.get_info(16440)}")
                    except cl.LogicError:
                        pass
                    print(f"          Version:          {d.get_info(cl.device_info.VERSION)}")
                    print(f"          Driver version:   {d.get_info(cl.device_info.DRIVER_VERSION)}")
                    print(f"          float64 support:  {'cl_khr_fp64' in d.get_info(cl.device_info.EXTENSIONS)}")
                    print(f"          float16 support:  {'cl_khr_fp16' in d.get_info(cl.device_info.EXTENSIONS)}")
