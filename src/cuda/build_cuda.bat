
:: cudart and nvrtc both need a specific version, so we can't rely on it being present on the system.
:: nvrtc can be downloaded from PyPi with a specific version so that's fine; cudart is statically linked so doesn't need external resolution.

call "vcvarsall.bat" x64
copy ..\vkfft_cuda.cu vkfft_cuda.cpp
cl /LD /MD /I..\VkFFT\vkFFT /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include"  vkfft_cuda.cpp /link /NODEFAULTLIB:LIBCMT /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64" nvrtc.lib cuda.lib cudart_static.lib