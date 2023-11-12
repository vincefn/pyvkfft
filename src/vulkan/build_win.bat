call "vcvars64" 
copy ..\vkfft_vulkan.cpp vkfft_vulkan.cpp
cl /LD /MD /EHsc /DVK_API_VERSION=11 /I..\VkFFT\vkFFT /I"C:\VulkanSDK\1.3.261.1\Include" /I"C:\VulkanSDK\1.3.261.1\Include\glslang\Include" vkfft_vulkan.cpp /link /LIBPATH:"C:\VulkanSDK\1.3.261.1\Lib" vulkan-1.lib GenericCodeGen.lib glslang.lib MachineIndependent.lib SPIRV.lib SPIRV-Tools.lib SPIRV-Tools-opt.lib
copy vkfft_vulkan.dll ..\..\..\radis\radis\gpu\vulkan\bin\vkfft_vulkan.dll