import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft,fftshift,ifftshift
import cmath
from matplotlib.ticker import LinearLocator

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



#id_tA = 2|A|^2*A+d2_xA
#-l<x<l, l = 10
c = 1
lam = 2
L = 20
#k = np.pi/L/c ## delta k, fft.shift


x_n, x_0 = L, -L
n_x = 100
h = (x_n - x_0)/(n_x-1)
t_0 = 0
n_t = 100
tau = 1/(n_t-1)
x_arr = np.array([x_0 + h*j for j in range(n_x)])
t_arr = np.array([t_0 + tau*i for i in range(n_t)])
A_0_arr = np.array([c*lam/np.cosh(lam*x) for x in x_arr])

k_arr = np.zeros(n_t)

for j in range(n_t):
    if j<n_t/2:
        k_arr[j] = j/n_x*2*np.pi*n_x/L/c
    else:
        k_arr[j] = (-1 + j/n_x)*2*np.pi*n_x/L/c

#𝐴(𝑧, 𝑡) = 𝐴(0, 𝑡)𝑒^𝑖κ𝑧
def Theory():
    solve = np.zeros((n_t,n_x))*1j
    intensity=np.zeros((n_t,n_x))
    for i in range(n_t):
        for j in range(n_x):
            solve[i][j] = cmath.exp(1j * t_arr[i] * 1) * A_0_arr[j]
            intensity[i][j] = abs(solve[i][j])**2
    return intensity

intens_theory=Theory()
plt.plot(x_arr,intens_theory[0][:],label= "Theory first moment")
plt.plot(x_arr,intens_theory[n_x-1][:],label = "Theory last moment")
plt.title("Theory")
plt.legend()
plt.show()

def step_i(A_0, tau, k):
    fi_i = np.array([cmath.exp((-1j)*tau*2*abs(A)**2)*A for A in A_0])
    ft_fi_i = fftshift(fi_i)
    ft_fi_i = ft_fi_i * np.array([cmath.exp((1j)*tau*k_**2) for k_ in k])
    F_=ifftshift(ft_fi_i)
    return F_

solution= 1j*np.zeros((n_t,n_x))
intensity = np.zeros((n_t,n_x))
solution[0] = A_0_arr
intensity[0] = abs(A_0_arr)**2
for i in range(1,n_t):
    solution[i] = (step_i(A_0 = solution[i-1], tau = tau, k = k_arr))
    intensity[i] = (abs(solution[i])**2)

solution = np.array(solution)
intensity = np.array(intensity)
################### GRAPH    ########################
fig = plt.figure()
ax = plt.axes(projection='3d')
t_arr_new,x_arr_new= np.meshgrid(t_arr,x_arr)
surf  = ax.plot_surface(t_arr_new, x_arr_new, intensity, cmap='inferno',
                          linewidth=6, antialiased=False,label="Calculate")
ax.text2D(0.05, 0.95, "Calculate", transform=ax.transAxes)
ax.set_xlabel("t")
ax.set_ylabel("x")
#ax.set_zscale('log')
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
#################   SECOND  ###################
fig = plt.figure()
ax = plt.axes(projection='3d')
t_arr_new,x_arr_new= np.meshgrid(t_arr,x_arr)
surf  = ax.plot_surface(t_arr_new, x_arr_new, intens_theory, cmap='viridis',
                          linewidth=6, antialiased=False ,label = "Theory")
ax.text2D(0.05, 0.95, "Theory", transform=ax.transAxes)
ax.set_xlabel("t")
ax.set_ylabel("x")
#ax.set_zscale('log')
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

####################    THIRD   #####################
plt.plot(x_arr, intensity[0][:],label="first moment")
plt.plot(x_arr, intensity[n_x-1][:],label="last moment",color = "red")
plt.title("Calculate")
plt.legend()
plt.show()
#Agrowal nonlinear photonic optics
#порядок метода


print("################### ERRORS #############################")
err1 = max(intensity[:][1] - intens_theory[:][1])
relatively_err1 = max((intensity[:][1] - intens_theory[:][1])/ intens_theory[:][1])
print("Максимальная Ошибка в первый момент времени",err1)
print("Максимальная Относительная Ошибка в первый момент времени",relatively_err1)
plt.plot(x_arr,intensity[0][:] - intens_theory[0][:])
plt.title("Ошибка в первый момент времени")
plt.show()
err2 = max(intensity[n_x-1][:] - intens_theory[n_x-1][:])
relatively_err2 = max((intensity[n_x-1][:] - intens_theory[n_x-1][:])/ intens_theory[n_x-1][:])
plt.plot(x_arr,intensity[n_x-1][:] - intens_theory[n_x-1][:])
plt.title("Ошибка во второй момент времени")
plt.show()
print("Максимальная Относительная Ошибка во второй момент времени",relatively_err2)
print("Максимальная Ошибка во второй момент времени",err2)