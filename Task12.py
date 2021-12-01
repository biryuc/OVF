from math import *

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft


#амплитуды
a0 = 1
a1 = 0.002

#частоты
w0 = 5.1
w1 = 5*w0

#интервал
T = 2*pi
t0 = 0
tn = T - t0

n = 100

#точки времени
t_s = np.linspace(t0, tn, n)


def f(t):
    return a0*sin(w0*t) + a1*sin(w1*t)

#################   ОКНА  №№№№№№№№№№№№№№№№№№
def rec_w(k):
    if k>=0 and k < n:
            return 1
    return 0

def hann_w(k):
    if k >= 0 and k < n:
        return 0.5*(1 - cos(2*pi*k/(n-1)))
    return 0
def n_w(k):
    return 1
###################     ####################

def fourie_pow_spect(f, h, t_s):
    pow_spect = []
    w = []
    for i in range(n):
        fi = complex(0, 0)

        for k in range(n):
            fi += (1 / n) * f(t_s[k]) * np.exp(2 * np.pi * (1j) * i * k / n) * h(k)

        pow_spect.append((fi * fi.conjugate()).real)  # мощность - квадрат модуля
        w.append(2 * np.pi * i / T)

    return w, pow_spect


w_no, spec_no = fourie_pow_spect(f, n_w, t_s)

#plt.figure(figsize=(15, 6))
# plt.plot(w_no, spec_no, 'mo-')
# plt.grid()
# plt.xlabel('w')
# plt.ylabel('|f(w)|^2')
# plt.title('Спектр мощность, без окна')
# plt.show()

w_rec, spec_rec = fourie_pow_spect(f, rec_w, t_s)

plt.figure(figsize=(15, 6))
plt.plot(w_rec, spec_rec, 'mo-')
plt.grid()
plt.xlabel('w')
plt.ylabel('|f(w)|^2')
plt.title('Спектр мощность, прямоугольное окно')
plt.show()

w_hann, spec_hann = fourie_pow_spect(f, hann_w, t_s)

plt.figure(figsize=(15, 6))
plt.plot(w_hann, spec_hann, 'mo-')
plt.grid()
plt.xlabel('w')
plt.ylabel('|f(w)|^2')
plt.title('Спектр мощность, окно Ханна')
plt.show()


def fourie_pow_spect_2(f, h, t_s):
    pow_spect = []
    w = []
    for i in range(round(n / 2)):
        fi = complex(0, 0)

        for k in range(n):
            fi += (1 / n) * f(t_s[k]) * np.exp(2 * np.pi * (1j) * i * k / n) * h(k)

        pow_spect.append((fi * fi.conjugate()).real)  # мощность - квадрат модуля
        w.append(2 * np.pi * i / T)

    return w, pow_spect



# w_no_2, spec_no_2 = fourie_pow_spect_2(f, no_w, t_s)
#
# plt.figure(figsize=(15, 6))
# plt.plot(w_no_2, spec_no_2, 'mo-')
# plt.grid()
# plt.xlabel('w')
# plt.yscale('log')
# plt.ylabel('|f(w)|^2')
# plt.title('Спектр мощность, без окна, лог. масштаб')
#
# plt.axvline(5.1, c='k', linestyle='dashed')
# plt.axvline(25.5, c='k', linestyle='dashed')
#
# plt.show()
#
#
# w_rec_2, spec_rec_2 = fourie_pow_spect_2(f, rec_w, t_s)
#
# plt.figure(figsize=(15, 6))
# plt.plot(w_rec_2, spec_rec_2, 'mo-')
# plt.grid()
# plt.xlabel('w')
# plt.ylabel('|f(w)|^2')
# plt.yscale('log')
# plt.title('Спектр мощность, прямоугольное окно, лог. масштаб')
#
# plt.axvline(5.1, c='k', linestyle='dashed')
# plt.axvline(25.5, c='k', linestyle='dashed')
#
# plt.show()
#
#
# w_hann_2, spec_hann_2 = fourie_pow_spect_2(f, hann_w, t_s)
#
# plt.figure(figsize=(15, 6))
# plt.plot(w_hann_2, spec_hann_2, 'mo-')
# plt.grid()
# plt.xlabel('w')
# plt.ylabel('|f(w)|^2')
# plt.yscale('log')
# plt.title('Спектр мощность, окно Ханна, лог. масштаб')
#
# plt.axvline(5.1, c='k', linestyle='dashed')
# plt.axvline(25.5, c='k', linestyle='dashed')
#
# plt.show()
func = [f(t) for t in t_s]
w = [2 * np.pi * i / T for i in range(n)]
spectr = fft(func)
def no_window():

    with_no_win = spectr
    spectr_no_win=[]
    for i in range(n):
        res = with_no_win[i]
        spectr_no_win.append((res*res.conjugate()).real/10000)
    plt.plot(w_no, spec_no, 'mo-')
    #plt.grid()
    plt.xlabel('w')
    plt.ylabel('|f(w)|^2')
    plt.title('Спектр мощность, без окна')
    plt.plot(w,spectr_no_win)
    plt.show()
    err = [spec_no[i] - spectr_no_win[i] for i in range(n)]
    plt.plot(w, err)
    plt.show()
def sq_window():
    func = [f(t)*rec_w(t) for t in t_s]
    #with_no_win = spectr
    spectr = fft(func)
    spectr_sq_win = []
    for i in range(n):
        res = spectr[i]
        spectr_sq_win.append((res * res.conjugate()).real / 10000)
    plt.plot(w_rec, spec_rec, 'x', label = "Рассчитанное")
    # plt.grid()
    plt.xlabel('w')
    plt.ylabel('|f(w)|^2')
    plt.title('Спектр мощность, прямоугольное окно')
    plt.plot(w, spectr_sq_win,label ="Теория")
    plt.legend()
    plt.show()
    err = [spec_no[i] - spectr_sq_win[i] for i in range(n)]
    plt.plot(w, err)
    plt.show()
def hann_window():
    func = [f(t)*hann_w(t) for t in t_s]
    spectr_hann = fft(func)
    spectr_hann_win = []
    for i in range(n):
        res = spectr_hann[i]
        spectr_hann_win.append((res * res.conjugate()).real)
    plt.plot(w_hann, spec_hann, 'o',label = "Рассчитанное")
    # plt.grid()
    plt.xlabel('w')
    plt.ylabel('|f(w)|^2')
    plt.title('Спектр мощность, окно Ханна')
    plt.plot(w_hann, spectr_hann_win,label ="Теория")
    plt.legend()
    plt.show()
    err = [spec_hann[i] - spectr_hann_win[i] for i in range(n)]
    plt.plot(w, err)
    plt.show()

no_window()
sq_window()
hann_window()


