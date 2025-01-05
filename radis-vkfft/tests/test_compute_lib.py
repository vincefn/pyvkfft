
from sys import path
path.append('..')
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



t_max = 100.0
Nt = 305
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
Nb = 10 # batch

params_h = params_t()
params_h.Nf = Nf
params_h.Nb = Nb
params_h.dt = dt
params_h.wL = w0


I_arr = mock_spectrum(Nt, Nl, m=Nb)
I_arr_FT = np.fft.rfft(I_arr)

# Init Vulkan
shader_path = os.path.dirname(__file__)
app = GPUApplication(deviceID=1, path=shader_path)


app.params_d = GPUStruct.fromStruct(params_h, binding=0)
app.I_arr_d = GPUArray.fromArr(I_arr, binding=1)
app.I_arr_FT_d = GPUArray((Nb, Nf), np.complex64, binding=2)
app.res_FT_d = GPUArray((Nb, Nf), np.complex64, binding=3)
app.res_d = GPUArray((Nb, Nt), np.float32,   binding=4)

##app.I_arr_d.setData(I_arr)

app.command_list = [
    app.cmdAddTimestamp("start"),
    # app.cmdClearBuffer(app.S_klm_d, timestamp=True),
    # app.cmdFillLDM((init_h.N_lines // N_tpb + 1, 1, 1), threads, timestamp=True),
    # app.cmdClearBuffer(app.S_klm_FT_d, timestamp=True),
    app.cmdFFT(app.I_arr_d, app.I_arr_FT_d, timestamp=True),
    # app.cmdClearBuffer(app.spectrum_FT_d, timestamp=True),
    #app.cmdApplyTestLineshape(
    #    (Nf * Nb // Ntpb + 1, 1, 1), threads, timestamp=True
    #),
    # app.cmdClearBuffer(app.spectrum_d, timestamp=True),
    #app.cmdIFFT(app.I_arr_FT_d, app.res_d, timestamp=True),
]
app.writeCommandBuffer()



def update(val):
    wL = sw.val

    # Recalc ref
    t0 = perf_counter()
    lineshape_FT = L_FT(f_arr, wL)
    I_arr_FT = np.fft.rfft(I_arr) #* lineshape_FT[np.newaxis,:]
##    res_ref = np.fft.irfft(I_arr_FT)
    res_ref = I_arr_FT.real   
    
    t1 = perf_counter()
    
    # Recalc gpu
    params_h.wL = wL
    app.params_d.setData(params_h)
    app.run()
##    res_gpu = np.copy(app.res_d.getData())
    res_gpu = np.copy(app.I_arr_FT_d.getData().real)
    t2 = perf_counter()

##    print('ref:',(t1-t0)*1e3)
##    print('gpu:',(t2-t1)*1e3)
    
    for i in range(Nb):
        lines_ref[i].set_ydata(res_ref[i] - res_gpu[i])
##        lines_gpu[i].set_ydata(res_gpu[i])
    fig.canvas.draw_idle()



# Set up plotting
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

ax.axhline(0, c="k", lw=1, alpha=0.5)
lines_ref = ax.plot(f_arr, np.zeros_like(I_arr_FT).real.T, '.', lw=1)
lines_gpu = ax.plot(f_arr, np.zeros_like(I_arr_FT).real.T, 'k--', lw=1)

axw = plt.axes([0.25, 0.1, 0.65, 0.03])
sw = Slider(axw, "Width", 0.0, 2.0, valinit=w0)
update(w0)
ax.relim()
ax.set_xlim(-0.05,0.2)
sw.on_changed(update)
plt.show()


