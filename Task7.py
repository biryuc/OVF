import numpy as np
import matplotlib.pyplot as plt
from math import *

a, b, c, d = 10, 2, 2, 10
leftbounds,rightbounds=0,5
predator,prey = 6, 6
alpha = 0.75
N = 1000

def f1(x,y):
    return a*x - b*x*y
def f2(x,y):
    return c*x*y - d*y

def RungeKuttaMethods( N=500,  alpha=3/4,  InitPointX=7,InitPointY=7,leftbounds=0,rightbounds=10):
    x=[InitPointX]
    y=[InitPointY]
    t=[leftbounds]
    h = (rightbounds - leftbounds) / N
    for i in range(N):
        t.append(t[i] + h)
        x.append(x[i] + h * ((1 - alpha) * f1(x[i], y[i]) + alpha * f1(x[i] + h / (2 * alpha) * f1(x[i], y[i]), y[i] + h / (2 * alpha) * f2(x[i], y[i]))))
        y.append(y[i] + h * ((1 - alpha) * f2(x[i], y[i]) + alpha * f2(x[i] + h / (2 * alpha) * f1(x[i], y[i]), y[i] + h / (2 * alpha) * f2(x[i], y[i]))))

    return t[1:],x[1:],y[1:]

def Show(plotter, name, data):
    plotter.set_title(name)
    plotter.set_xlabel('t')
    plotter.set_ylabel('X and Y')
    plotter.plot(data[0], data[1], label="X(t)")
    plotter.plot(data[0], data[2], label="Y(t)")

def Show2(plotter, data):
    plotter.set_title("Y(X)")
    plotter.plot(data[1], data[2], label="Y(X)")
if __name__ == '__main__':
    fig = plt.figure(figsize=(8, 4))
    plt_x = fig.add_subplot(1, 2, 1)
    plt2 = fig.add_subplot(1, 2, 2)
    Show(plt_x, 'X(t),Y(t)', RungeKuttaMethods())
    plt_x.legend()
    Show2(plt2,RungeKuttaMethods())
    plt2.legend()
    plt.show()
    N = [i * 100 for i in range(1, 21)]
    h = [(rightbounds - leftbounds) / n for n in N]
    for i in range(len(N)):
        print(N[i], ':', h[i])



