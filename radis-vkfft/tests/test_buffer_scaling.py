
import sys
sys.path.append('..')
import os

from vulkan_compute_lib import GPUApplication, GPUArray, GPUStruct
import numpy as np
from scipy.fft import next_fast_len
from numpy.fft import rfftfreq
from numpy.random import rand, randint, seed
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from ctypes import Structure, c_float, c_int
from time import perf_counter
import vulkan as vk

L = lambda t, w: 2 / (w * np.pi) * 1 / (1 + 4 * (t / w) ** 2)
L_FT = lambda f, w: np.exp(-np.pi * np.abs(f) * w)

class params_t(Structure):
    _fields_ = [
        ("Nf", c_int),
        ("Nb", c_int),
        ("dt", c_float),
        ("wL", c_float),
    ]

def next_fast_len_even(n):
    n = next_fast_len(n)
    while n&1:
        n = next_fast_len(n+1)
    return n


def mock_spectrum(Nt, Nl, m=1):
    
    I_arr = np.zeros((m, Nt), dtype=np.float32)

    for k in range(m):
        line_index = randint(0, Nt, Nl)
        line_strength = rand(Nl)
        np.add.at(I_arr, (k, line_index), line_strength)
    return I_arr


device_id = 2
t_max = 100.0
Nt = 25600
Nt = next_fast_len_even(Nt)
print("Nt = {:d}".format(Nt))
t_arr = np.linspace(0, t_max, Nt)
dt = t_arr[1]

##f_arr = rfftfreq(Nt, dt)
##Nf = len(f_arr)

Nf = Nt//2 + 1
f_arr = np.arange(Nf) / (Nf * dt)

print(Nf)


Ntpb = 1024  # threads per block
threads = (Ntpb, 1, 1)
w0 = 0.0
seed(1)
Nl = 200
Nb_max = 10
Nb = 5 # batch

params_h = params_t()
params_h.Nf = Nf
params_h.Nb = Nb
params_h.dt = dt
params_h.wL = w0


I_arr = mock_spectrum(Nt, Nl, m=Nb_max)
I_arr_FT = np.fft.rfft(I_arr).real

# Init Vulkan
shader_path = os.path.dirname(__file__)
app = GPUApplication(deviceID=device_id, path=shader_path)
app.print_memory_properties()

props_device = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
props_host = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
props_mixed = props_device | props_host

if (memtype := app.findMemoryType3(31, props_mixed)) < 0:
    memtype = (app.findMemoryType3(31, props_device),
               app.findMemoryType3(31, props_host))

print(memtype)
sys.exit()

app.params_d = GPUStruct.fromStruct(params_h, binding=0)
##app.I_arr_d = GPUArray.fromArr(I_arr, binding=1)
app.I_arr_d = GPUArray((Nb_max, Nt), np.float32, binding=1)
app.I_arr_FT_d = GPUArray((Nb, Nf), np.complex64, binding=2)
app.res_FT_d = GPUArray((Nb, Nf), np.complex64, binding=3)
app.res_d = GPUArray((Nb, Nt), np.float32,   binding=4)

app.I_arr_d.setData(I_arr)

app.command_list = [
    # app.cmdAddTimestamp("start"),
    app.cmdClearBuffer(app.I_arr_FT_d, timestamp=False),
    # app.cmdFillLDM((init_h.N_lines // N_tpb + 1, 1, 1), threads, timestamp=True),
    # app.cmdClearBuffer(app.S_klm_FT_d, timestamp=True),
    app.cmdFFT(app.I_arr_d, app.I_arr_FT_d, timestamp=False),
    # app.cmdClearBuffer(app.spectrum_FT_d, timestamp=True),
    #app.cmdApplyTestLineshape(
    #    (Nf * Nb // Ntpb + 1, 1, 1), threads, timestamp=True
    #),
    # app.cmdClearBuffer(app.spectrum_d, timestamp=True),
    #app.cmdIFFT(app.I_arr_FT_d, app.res_d, timestamp=True),
]
app.writeCommandBuffer()



def update(val):
    Nb = sN.val

    # Shuffle mock spectral lines:
##    I_arr = mock_spectrum(Nt, Nl, m=Nb)
##    app.I_arr_d.reshape((Nb, Nt))
##    app.I_arr_d.setData(I_arr)

    # Recalc ref
    res_ref = np.zeros((Nb_max, Nf), dtype=np.float32)
    I_arr_FT = np.fft.rfft(I_arr[:Nb,:]).real
    res_ref[:Nb] = I_arr_FT[:Nb]

    # Recalc gpu
    params_h.Nb = Nb
    app.params_d.setData(params_h)
    

    app.I_arr_FT_d.reshape((Nb, Nf))

    #app.freeCommandBuffer()
    app.writeCommandBuffer()


    
    t0 = perf_counter()
    print('running command buffer...')
    app.run()
    
    res_gpu = app.I_arr_FT_d.getData().real
    #print(res_gpu.shape)
    t1 = perf_counter()

    ax.set_title(f'{(t1-t0)*1e3:.2f} ms')
    
    for i in range(Nb_max):        
        lines_ref[i].set_ydata(res_ref[i])
        if i < Nb:
            lines_gpu[i].set_ydata(res_gpu[i])
        else:
            lines_gpu[i].set_ydata(np.zeros(Nf))
    fig.canvas.draw_idle()


# Set up plotting
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

ax.axhline(0, c="k", lw=1, alpha=0.5)
lines_ref = ax.plot(f_arr, np.zeros((Nb_max, Nf), dtype=np.float32).real.T, '-', lw=1)
lines_gpu = ax.plot(f_arr, np.zeros((Nb_max, Nf), dtype=np.float32).real.T, '.', lw=1)

axN = plt.axes([0.25, 0.1, 0.65, 0.03])
axw = plt.axes([0.25, 0.05, 0.65, 0.03])
sN = Slider(axN, "N batch", 1, Nb_max, valstep=1, valinit=Nb)
sw = Slider(axw, "randomize", 1, Nb_max, valstep=1, valinit=Nb)

update(Nb)
ax.relim()
##ax.set_xlim(2.2,2.6)
sN.on_changed(update)
sw.on_changed(update)
plt.show()


