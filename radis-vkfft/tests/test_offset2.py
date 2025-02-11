
from sys import path
path.append('..')
import os

from vulkan_compute_lib import GPUApplication, GPUBuffer
import numpy as np
from scipy.fft import next_fast_len
from numpy.fft import rfftfreq
from numpy.random import rand, randint, seed
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from ctypes import Structure, c_float, c_int, sizeof
from time import perf_counter
import vulkan as vk
import sys

L = lambda t, w: 2 / (w * np.pi) * 1 / (1 + 4 * (t / w) ** 2)
L_FT = lambda f, w: np.exp(-np.pi * np.abs(f) * w)

class init_params_t(Structure):
    _fields_ = [
        ("Nl", c_int),
        ("Nt", c_int),
        ("Nf", c_int),

        ("t_min", c_float),
        ("dt",    c_float),
        ("w_min", c_float),
    ]

class iter_params_t(Structure):
    _fields_ = [
        ("a", c_float),
        ("Nw", c_int),
        ("dxw",   c_float),
    ]



def next_fast_len_even(n):
    n = next_fast_len(n)
    while n&1:
        n = next_fast_len(n+1)
    return n


def mock_spectrum(Nl, t_min=0.0, t_max=1.0, w_min=0.0, w_max=1.0):
    I0_arr = rand(Nl)
    t0_arr = rand(Nl)*(t_max - t_min) + t_min
    w0_arr = rand(Nl)*(w_max - w_min) + w_min

    return I0_arr.astype(np.float32), t0_arr.astype(np.float32), w0_arr.astype(np.float32)


t_min = 0.0
t_max = 100.0
Nt = 300001
Nt = next_fast_len_even(Nt)
print("Nt = {:d}".format(Nt))
t_arr = np.linspace(t_min, t_max, Nt)
dt = t_arr[1]

#f_arr = rfftfreq(Nt, dt)
#Nf = len(f_arr)

Nf = Nt//2 + 1
f_arr = np.arange(Nf) / (2 * Nf * dt)

print(Nf)


Ntpb = 1024  # threads per block
threads = (Ntpb, 1, 1)
w0 = 0.0
seed(1)
Nl = 2000
Nw = 8

w_min = 0.1
w_max = 3.0
dw = (w_max - w_min) / (Nw - 1)
dxw = np.log(w_max / w_min) / (Nw - 1)


I0_arr, t0_arr, w0_arr = mock_spectrum(Nl, t_min=t_min, t_max=t_max, w_min=w_min, w_max=w_max)
database = np.array([I0_arr, t0_arr, w0_arr])


I_arr = I0_arr

Nfwd = I0_arr.shape[0]
Nrev = Nfwd//2+1
Nrev2 = 2*Nrev



I_arr_FT = np.zeros(2*Nrev2, dtype=np.float32)
I_arr_FT[:Nfwd] = I0_arr*0.5+0.5
I_arr_FT[Nrev2:Nrev2+Nfwd] = I0_arr*0.7+0.7

I_arr = np.zeros_like(I_arr_FT)
I_arr[:] = I_arr_FT


#I_arr_FT[:Nrev2] = np.fft.rfft(I_arr_FT[:Nfwd]).view(np.float32)
I_arr_FT[Nrev2:] = np.fft.rfft(I_arr_FT[Nrev2:Nrev2+Nfwd]).view(np.float32)




 
#%% GPU vulkan
print('GPU start...')
shader_path = os.path.dirname(__file__)
app = GPUApplication(deviceID=1, path=shader_path)
#app.print_memory_properties()

# # Reuse buffer for input and output

app.spectrum_d = GPUBuffer(2*Nrev2*4)

app.spectrum_d.setFFTShape(Nfwd, np.float32)
# #app.spectrum_FT_d.setFFTShape(Nrev2, np.complex64)

app.spectrum_d.initStagingBuffer()
# #app.spectrum2_d.initStagingBuffer()
# #app.spectrum_FT_d.initStagingBuffer()


app.command_list = [
    #app.cmdClearBuffer(app.spectrum_FT_d),
    app.spectrum_d.cmdTransferStagingBuffer('H2D'),
    app.cmdFFT(app.spectrum_d, app.spectrum_d, 0, Nrev2*4),
    app.spectrum_d.cmdTransferStagingBuffer('D2H'),
]
app.writeCommandBuffer()

app.spectrum_d.fromArray(I_arr)
app.run()

I_arr_FT2 = np.zeros_like(I_arr_FT)
app.spectrum_d.toArray(I_arr_FT2)


plt.plot(I_arr_FT)
plt.plot(I_arr_FT2,'k--')
sys.exit()
# Works!

















#%%

# #ax.plot(t_arr, I_arr0)

# #p1 = ax.plot(f_arr, S_kl_FT.T.real)
# #p2 = ax.plot(f_arr, S_kl_FT2.T.real, 'k--')
# p1, = ax.plot(t_arr, I_arr1)
# p2, = ax.plot(t_arr, I_arr2, 'k--')

# axNw = plt.axes([0.25, 0.05, 0.65, 0.03])
# sNw = Slider(axNw, "Nw", 2, 20, valinit=Nw, valstep=1)

# axw = plt.axes([0.25, 0.1, 0.65, 0.03])
# sw = Slider(axw, "a", -1.0, 2.0, valinit=0.0)

# Nw_i = Nw
# def update(val):
#     global Nw_i
    

#     a = sw.val

#     t0 = perf_counter()
#     I_arr1 = spectrum_dit(a)
#     t1 = perf_counter()

#     if sNw.val != Nw_i:
#         #print('new val', sNw.val)
#         Nw_i = sNw.val
#         dxw_i = np.log(w_max / w_min) / (Nw_i - 1)
#         iter_params_h.Nw = Nw_i
#         iter_params_h.dxw = dxw_i
#         app.S_kl_d.setFFTShape((Nw_i+1, Nt))
#         app.S_kl_FT_d.setFFTShape((Nw_i+1, Nf))
#         #app.updateDescriptorSet(app._descriptorSets[0])

#         app.freeCommandBuffer()
#         app.writeCommandBuffer()
        

    
#     iter_params_h.a = a
#     app.run()
#     app.spectrum_d.toArray(I_arr2)
#     t2 = perf_counter()

#     ax.set_title('CPU: {:.1f} ms - GPU: {:.1f} ms'.format((t1-t0)*1e3, (t2-t1)*1e3))
#     p1.set_ydata(I_arr1)
#     p2.set_ydata(I_arr2)
    
#     fig.canvas.draw_idle()

# sw.on_changed(update)
# sNw.on_changed(update)

# plt.show()



