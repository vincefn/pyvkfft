
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
        ("Nl", c_int),
        ("Nt", c_int),
        ("Nf", c_int),
        ("Nw", c_int),

        ("t_min", c_float),
        ("dt",    c_float),
        ("w_min", c_float),
        ("dxw",   c_float),
    ]

class iter_params_t(Structure):
    _fields_ = [
        ("a", c_float),
        ("b", c_float),
        ("c", c_float),
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
Nt = 3001
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
Nl = 200
Nw = 8

w_min = 0.1
w_max = 3.0
dw = (w_max - w_min) / (Nw - 1)
dxw = np.log(w_max / w_min) / (Nw - 1)



I0_arr, t0_arr, w0_arr = mock_spectrum(Nl, t_min=t_min, t_max=t_max, w_min=w_min, w_max=w_max)
database = np.array([I0_arr, t0_arr, w0_arr])

#%% CPU Legacy method:
print('Adding... ', end='')
tc0 = perf_counter()
I_arr0 = np.zeros(Nt, dtype=np.float32)
for I0, t0, w0 in database.T:
    I_arr0 += I0*L(t_arr - t0, w0)
tc1 = perf_counter()
print('Done! {:.3f}'.format((tc1-tc0)*1e3))


#%% CPU DIT method:
# lineshape distribution:
print('Distributing... ', end='')
tc0 = perf_counter()
k_arr = (database[1] - t_min) / dt
k0_arr = k_arr.astype(np.int32)
k1_arr = k0_arr + 1
a1k_arr = k_arr - k0_arr
a0k_arr = 1 - a1k_arr

l_arr = (np.log(database[2]) - np.log(w_min)) / dxw
l0_arr = l_arr.astype(np.int32)
l1_arr = l0_arr + 1
a1l_arr = l_arr - l0_arr
a0l_arr = 1 - a1l_arr

a00_arr = a0l_arr * a0k_arr
a01_arr = a0l_arr * a1k_arr
a10_arr = a1l_arr * a0k_arr
a11_arr = a1l_arr * a1k_arr

S_kl = np.zeros((Nw, Nt), dtype=np.float32)

np.add.at(S_kl, (l0_arr, k0_arr), database[0] * a00_arr)
np.add.at(S_kl, (l0_arr, k1_arr), database[0] * a01_arr)
np.add.at(S_kl, (l1_arr, k0_arr), database[0] * a10_arr)
np.add.at(S_kl, (l1_arr, k1_arr), database[0] * a11_arr)
#print('Done!')

#print('Applying lineshape... ', end='')
I_arr1 = np.zeros(Nt, dtype=np.float32)
S_kl_FT = np.fft.rfft(S_kl)
for l in range(Nw):
    w_l = w_min*np.exp(l*dxw)

    #S_k_FT = np.fft.rfft(S_kl[l])
    S_k_FT = S_kl_FT[l]
    #ls = L(t_arr - t_min - Nt/2*dt, w_l)
    #ls_FT = np.fft.rfft(np.fft.fftshift(ls))
    ls_FT = L_FT(f_arr, w_l)/dt
    I_k_FT = S_k_FT * ls_FT
    I_arr1 += np.fft.irfft(I_k_FT)
    #I_arr1 += np.convolve(S_kl[l], ls, 'same')

tc1 = perf_counter()
print('Done! {:.3f}'.format((tc1-tc0)*1e3))


#%% GPU vulkan
shader_path = os.path.dirname(__file__)
app = GPUApplication(deviceID=0, path=shader_path)
#app.print_memory_properties()

I_arr2 = np.zeros(Nt, dtype=np.int32)

app.init_params_d = GPUBuffer(sizeof(init_params_t), uniform=True, binding=0)
app.iter_params_d = GPUBuffer(sizeof(iter_params_t), uniform=True, binding=1)
app.database_d = GPUBuffer(database.nbytes, binding=2)
#app.database2_d = GPUBuffer(database.nbytes, binding=3)


app.S_kl_d = GPUBuffer(S_kl.nbytes, binding=3)
app.S_kl_FT_d = GPUBuffer(S_kl_FT.nbytes, binding=4)
app.spectrum_FT_d = GPUBuffer(I_k_FT.nbytes, binding=5)
app.spectrum_d = GPUBuffer(I_arr2.nbytes, binding=6)


# initalize data:
app.database_d.initStagingBuffer()
#app.database2_d.initStagingBuffer()
app.database_d.copyToBuffer(database)

app.init_params_d.initStagingBuffer()
init_params_h = app.init_params_d.getHostStructPtr(init_params_t)
init_params_h.Nl = Nl
init_params_h.Nt = Nt
init_params_h.Nf = Nf
init_params_h.Nw = Nw
init_params_h.t_min = t_min
init_params_h.dt = dt
init_params_h.w_min = w_min
init_params_h.dxw = dxw
app.init_params_d.transferStagingBuffer(direction='H2D')

app.iter_params_d.initStagingBuffer()
iter_params_h = app.iter_params_d.getHostStructPtr(iter_params_t)

app.S_kl_d.initStagingBuffer()
app.S_kl_d.setFFTShape((Nw,Nt), np.float32)
app.S_kl_FT_d.setFFTShape((Nw,Nf), np.complex64)
app.spectrum_FT_d.setFFTShape(Nf, np.complex64)
app.spectrum_d.setFFTShape(Nt, np.float32)

app.spectrum_d.initStagingBuffer()

app.command_list = [
    app.iter_params_d.cmdTransferStagingBuffer('H2D'),
    app.cmdClearBuffer(app.S_kl_d),
    app.cmdTestFillLDM((Nl // Ntpb + 1, 1, 1), threads),
    #app.cmdClearBuffer(app.S_kl_FT_d),
    #app.cmdFFT(app.S_kl_d, app.S_kl_FT_d),
    #app.cmdClearBuffer(app.spectrum_FT_d),
    #app.cmdTestApplyLineshapes((Nf // Ntpb + 1, 1, 1), threads),
    #app.cmdClearBuffer(app.spectrum_d),
    #app.cmdIFFT(app.spectrum_FT_d, app.spectrum_d), 
    #app.spectrum_d.cmdTransferStagingBuffer('D2H'),
]
app.writeCommandBuffer()


# iteration:

iter_params_h.a = 1000.0
iter_params_h.b = 100.0
iter_params_h.c = 1.0
app.run()
app.spectrum_d.toArray(I_arr2)

S_kl2 = np.zeros_like(S_kl)
app.S_kl_d.copyFromBuffer(S_kl2)
plt.plot(S_kl.T)
plt.plot(S_kl2.T,'k--')
plt.show()
# plt.plot(t_arr, I_arr2, 'r-.')






### command buffer:
##app.command_list = [
##    # app.cmdAddTimestamp("start"),
##    app.cmdClearBuffer(app.S_kl_d),
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








#%% Plotting:
# plt.plot(t_arr, I_arr0)
# plt.plot(t_arr, I_arr1, 'k--')
# plt.show()

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
