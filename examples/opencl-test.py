import numpy as np
import pyopencl as cl
import pyopencl.array as cla
from pyvkfft.opencl import VkFFTApp

ctx = cl.create_some_context()
cq = cl.CommandQueue(ctx)
a = cla.zeros(cq, (1, 1024, 3 * 5 * 7 * 11), np.complex64)
b = cla.empty_like(a)

app = VkFFTApp(a.shape, a.dtype, cq, ndim=2, inplace=True)

print(a.get().sum())
app.fft(a)
print("#" * 20, 'iFFT')
app.ifft(a)
print(a.get().sum())

app = VkFFTApp(a.shape, a.dtype, cq, ndim=2, inplace=False)

print(a.get().sum())
app.fft(a, b)
print("#" * 20, 'iFFT')
app.ifft(b, a)
print(a.get().sum())
