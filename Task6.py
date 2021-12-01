import numpy as np
import matplotlib.pyplot as plt
from math import *
N = 100
min, max = 0, 3
x0, t0 = 1, 0
h = (max - min)/N
alpha = 1/2

def X(x):
    return -x

def T(t):
    return 1/np.exp(t)

def Eiler():
    x, t = [x0], [t0]
    for i in range(N +1 ):
        x.append(x[i] + h * X(x[i]))
        t.append(t0 + h * i)
    return x[1:], t[1:]


def RungeKutta_2():
    x, t = [x0], [t0]
    for i in range(N +1):
        x.append(x[i] + h * ((1 - alpha) * X(x[i]) + alpha * X(x[i] + h * X(x[i]) / (2 * alpha))))
        t.append(t0 + h * i)
    return x[1:], t[1:]


def RungeKutta_4():
    x, t = [x0], [t0]
    for i in range(N + 1):
        k1 = X(x[i])
        k2 = X(x[i] + h * k1/2)
        k3 = X(x[i] + h * k2/2)
        k4 = X(x[i] + h * k3)
        x.append(x[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
        t.append(t0 + h * i)
    return x[1:], t[1:]
def solution():
    x_analyt, t = [x0], [t0]
    x=[x0]
    err=[x0]
    for i in range(N+1):
        x.append(x[i] + h * X(x[i]))
        t.append(t0 + h * i)
        x_analyt.append(np.exp(-t[i]))
        err.append(x[i]-x_analyt[i])
    return err[1:],t[1:]

def Err():
    x_analyt, t = [x0], [t0]
    for i in range(N + 1):
        t.append(t0 + h * i)
        x_analyt.append(np.exp(-t[i]))
    return x_analyt[1:], t[1:]
def Show(plotter, name, data,data1,data2,data3):
    plotter.set_title(name)
    plotter.set_xlabel('X')
    plotter.set_ylabel('T')
    plotter.plot(data[1], data[0],label="Solution")
    plotter.plot(data1[1], data1[0],label="Eiler")
    plotter.plot(data2[1], data2[0],label="2")
    plotter.plot(data3[1], data3[0],label="4")
def Showerr(plotter, name, data,data1):
    plotter.set_title(name)
    plotter.set_xlabel('X')
    plotter.set_ylabel('T')
    plotter.plot(data[1] - data1[1], data[0]-data[0],)

if __name__ == '__main__':
    t = np.arange(min, max, h)

    fig = plt.figure(figsize=(8, 4))
    plt_x = fig.add_subplot(1, 1, 1)

    fig.set_facecolor('green')
    Show(plt_x, 'Solution', (T(t), t),Eiler(),RungeKutta_2(),RungeKutta_4())
    plt.legend()
    plt.show()
    # Show(plt_euler, 'Euler', Eiler())
    # plt_rk2 = fig.add_subplot(2, 2, 3)
    # plt_rk4 = fig.add_subplot(2, 2, 4)
    # fig.set_facecolor('orange')
    # Show(plt_rk2, 'RK2', RungeKutta_2())
    # Show(plt_rk4, 'RK4', RungeKutta_4())
    # plt.show()
    # fig = plt.figure(figsize=(14, 7))
    # err = fig.add_subplot(1, 2, 1)
    # Show(err,"Error",solution())
    # plt.show()
