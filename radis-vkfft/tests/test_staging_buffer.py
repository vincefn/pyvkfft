
import sys
sys.path.append('..')
import os

from vulkan_compute_lib import GPUApplication, GPUArray, GPUStruct, GPUBuffer
import numpy as np
from scipy.fft import next_fast_len
from numpy.fft import rfftfreq
from numpy.random import rand, randint, seed
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from ctypes import Structure, c_float, c_int
from time import perf_counter
import vulkan as vk

# Init Vulkan
shader_path = os.path.dirname(__file__)
app = GPUApplication(deviceID=1, path=shader_path)
app.print_memory_properties()

#
N = 1024*1024
arr = np.arange(0, N, 2, dtype=np.int32)
arr2 = np.zeros(arr.size, dtype=np.int32)
arr3 = np.zeros(arr.size, dtype=np.int32)

print('arr.nbytes', arr.nbytes)

buf_dev = GPUBuffer(arr.nbytes, app=app)
buf_dev.initStagingBuffer(32*1024*1024)

buf_dev2 = GPUBuffer(arr.nbytes, app=app)
buf_dev2.initStagingBuffer(32*1024*1024)


print('Copying data... ')
print(arr)
t0 = perf_counter()
buf_dev.copyToBuffer(arr)
t1 = perf_counter()
buf_dev.copyFromBuffer(arr2)
##t2 = perf_counter()
print(arr2)
print('copy_to:  ', (t1-t0)*1000)

copy_region = vk.VkBufferCopy(srcOffset=0, dstOffset=0, size=buf_dev._bufferSize)
app.oneTimeCommand(vk.vkCmdCopyBuffer,
                   srcBuffer=buf_dev._buffer,
                   dstBuffer=buf_dev2._buffer,
                   regionCount=1,
                   pRegions=[copy_region],
                   )

t3 = perf_counter()
buf_dev2.copyFromBuffer(arr3)
t4 = perf_counter()
print(arr3)
print('copy_from:', (t4-t3)*1000)

print('Done!')
