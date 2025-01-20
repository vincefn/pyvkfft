
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

L = lambda t, w: 2 / (w * np.pi) * 1 / (1 + 4 * (t / w) ** 2)
L_FT = lambda f, w: np.exp(-np.pi * np.abs(f) * w)

class init_params_t(Structure):
    _fields_ = [
        ("Nt", c_int),
        ("Nf", c_int),
        ("Nw", c_int),
        ("Nl", c_int),
    ]

class iter_params_t(Structure):
    _fields_ = [
        ("A", c_int),
        ("B", c_int),
        ("C", c_int),
        #("dt", c_float),
        #("wL", c_float),
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

    return I0_arr, t0_arr, w0_arr


t_min = 0.0
t_max = 100.0
Nt = 3001
Nt = next_fast_len_even(Nt)
print("Nt = {:d}".format(Nt))
t_arr = np.linspace(t_min, t_max, Nt)
dt = t_arr[1]

f_arr = rfftfreq(Nt, dt)
Nf = len(f_arr)

#Nf = Nt//2 + 1
#f_arr = np.arange(Nf) / (Nf * dt)

print(Nf)


Ntpb = 1024  # threads per block
threads = (Ntpb, 1, 1)
w0 = 0.0
seed(1)
Nl = 200
Nw = 8

w_min = 0.1
w_max = 3.0
dw = (w_max - w_min) / (Nw - 1)
dxw = np.log(w_max / w_min) / (Nw - 1)



I0_arr, t0_arr, w0_arr = mock_spectrum(Nl, t_min=t_min, t_max=t_max, w_min=w_min, w_max=w_max)
database = np.array([I0_arr, t0_arr, w0_arr])


S_kl = np.zeros((Nw, Nt), dtype=np.float32)
S_kl_FT = np.fft.rfft(S_kl)
S_k_FT = S_kl_FT[0]
I_arr = np.zeros(Nt, dtype=np.int32)


#%% GPU vulkan
shader_path = os.path.dirname(__file__)
app = GPUApplication(deviceID=0, path=shader_path)
#app.print_memory_properties()

app.init_params_d = GPUBuffer(sizeof(init_params_t), uniform=True, binding=0)
app.iter_params_d = GPUBuffer(sizeof(iter_params_t), uniform=True, binding=1)
app.database_d = GPUBuffer(database.nbytes, binding=2)
app.S_kl_d = GPUBuffer(S_kl.nbytes, binding=3)
app.S_kl_FT_d = GPUBuffer(S_kl_FT.nbytes, binding=4)
app.spectrum_FT_d = GPUBuffer(S_k_FT.nbytes, binding=5)
app.spectrum_d = GPUBuffer(I_arr.nbytes, binding=6)

# initalize data:
app.init_params_d.initStagingBuffer()
init_params_h = app.init_params_d.getHostStructPtr(init_params_t)
init_params_h.Nt = Nt
init_params_h.Nf = Nf
init_params_h.Nw = Nw
init_params_h.Nl = Nl
app.init_params_d.transferStagingBuffer(direction='H2D')

app.iter_params_d.initStagingBuffer()
iter_params_h = app.iter_params_d.getHostStructPtr(iter_params_t)

app.database_d.initStagingBuffer(32*1024*1024)
app.database_d.copyToBuffer(database)

app.spectrum_d.initStagingBuffer()

app.command_list = [
    app.iter_params_d.cmdTransferStagingBuffer(direction='H2D'),
    app.cmdTestShader1((Nt // Ntpb + 1, 1, 1), threads),
    app.spectrum_d.cmdTransferStagingBuffer(direction='D2H'),
]
app.writeCommandBuffer()

# iteration:
iter_params_h.A = 2000
iter_params_h.B = 100
iter_params_h.C = 1

app.run()
app.spectrum_d.toArray(I_arr)
print(I_arr[:20])

iter_params_h.A = 3000
app.run()
app.spectrum_d.toArray(I_arr)
print(I_arr[:20])










##
###%% Plotting:
##plt.plot(t_arr, I_arr0)
##plt.plot(t_arr, I_arr1, 'k--')
##plt.show()


#########################################



##I_arr_FT = np.fft.rfft(I_arr)
##
### Init Vulkan
##shader_path = os.path.dirname(__file__)
##app = GPUApplication(deviceID=1, path=shader_path)
##
##
##app.params_d = GPUStruct.fromStruct(params_h, binding=0)
##app.I_arr_d = GPUArray.fromArr(I_arr, binding=1)
##app.I_arr_FT_d = GPUArray((Nb, Nf), np.complex64, binding=2)
##app.res_FT_d = GPUArray((Nb, Nf), np.complex64, binding=3)
##app.res_d = GPUArray((Nb, Nt), np.float32,   binding=4)
##
####app.I_arr_d.setData(I_arr)
##
##app.command_list = [
##    app.cmdAddTimestamp("start"),
##    # app.cmdClearBuffer(app.S_klm_d, timestamp=True),
##    # app.cmdFillLDM((init_h.N_lines // N_tpb + 1, 1, 1), threads, timestamp=True),
##    # app.cmdClearBuffer(app.S_klm_FT_d, timestamp=True),
##    app.cmdFFT(app.I_arr_d, app.I_arr_FT_d, timestamp=True),
##    # app.cmdClearBuffer(app.spectrum_FT_d, timestamp=True),
##    #app.cmdApplyTestLineshape(
##    #    (Nf * Nb // Ntpb + 1, 1, 1), threads, timestamp=True
##    #),
##    # app.cmdClearBuffer(app.spectrum_d, timestamp=True),
##    #app.cmdIFFT(app.I_arr_FT_d, app.res_d, timestamp=True),
##]
##app.writeCommandBuffer()
##
##
##
##def update(val):
##    wL = sw.val
##
##    # Recalc ref
##    t0 = perf_counter()
##    lineshape_FT = L_FT(f_arr, wL)
##    I_arr_FT = np.fft.rfft(I_arr) #* lineshape_FT[np.newaxis,:]
####    res_ref = np.fft.irfft(I_arr_FT)
##    res_ref = I_arr_FT.real   
##    
##    t1 = perf_counter()
##    
##    # Recalc gpu
##    params_h.wL = wL
##    app.params_d.setData(params_h)
##    app.run()
####    res_gpu = np.copy(app.res_d.getData())
##    res_gpu = np.copy(app.I_arr_FT_d.getData().real)
##    t2 = perf_counter()
##
####    print('ref:',(t1-t0)*1e3)
####    print('gpu:',(t2-t1)*1e3)
##    
##    for i in range(Nb):
##        lines_ref[i].set_ydata(res_ref[i] - res_gpu[i])
####        lines_gpu[i].set_ydata(res_gpu[i])
##    fig.canvas.draw_idle()
##
##
##
### Set up plotting
##fig, ax = plt.subplots()
##plt.subplots_adjust(left=0.25, bottom=0.25)
##
##ax.axhline(0, c="k", lw=1, alpha=0.5)
##lines_ref = ax.plot(f_arr, np.zeros_like(I_arr_FT).real.T, '.', lw=1)
##lines_gpu = ax.plot(f_arr, np.zeros_like(I_arr_FT).real.T, 'k--', lw=1)
##
##axw = plt.axes([0.25, 0.1, 0.65, 0.03])
##sw = Slider(axw, "Width", 0.0, 2.0, valinit=w0)
##update(w0)
##ax.relim()
##ax.set_xlim(-0.05,0.2)
##sw.on_changed(update)
##plt.show()
##
##
