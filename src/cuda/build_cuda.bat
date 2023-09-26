call "vcvarsall" x64
copy ..\vkfft_cuda.cu vkfft_cuda.cpp
cl /LD /MD /I..\VkFFT\vkFFT /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include"  vkfft_cuda.cpp /link /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64" nvrtc.lib cuda.lib cudart.lib