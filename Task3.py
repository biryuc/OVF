from math import *
import numpy as np
import matplotlib.pyplot as plt
def integral_1(x):
    function = 1/(1+pow(x,2))       #1.5708
    return function
def integral_2(x):
    function = x**(1/3)*exp(sin(x))
    return function

def massiveX(N):
    n=int(N/2-1)
    x = np.zeros(n)
    for i in range(n):
        x[i]=i
    return x
def massiveX_1(N):
    x = np.zeros(N-1)
    for i in range(N-1):
        x[i]=i
    return x
def trapezoid_rule(func, a, b, nseg,show):
    sum = 0.5 * (func(a) + func(b))
    arr = []
    last_sum=0
    dx = 1.0 * (b - a) / nseg
    for i in range(1, nseg):
        last_sum = sum
        sum += func(a + i * dx)
        arr.append(sum * dx)
        if i%2==0 and show==1:
            print("Iteration\t",i)
            if func == integral_1:
                print("Относительная Ошибка численного интегрирования методом трапеций для первого интеграла =", (pi/2 - sum*dx)/(pi/2))
            else:
                print("Относительная Ошибка численного интегрирования методом трапеций для второго интеграла =",abs((1.29587400873 - sum * dx))/ (1.29587400873))
    return arr

def simpson_rule(func, a, b, nseg,show):
    if nseg%2 == 1:
       nseg += 1
    dx = 1.0 * (b - a) / nseg
    arr=[]
    sum = (func(a) + 4 * func(a + dx) + func(b))
    for i in range(1, int(nseg*0.5)):
        last_sum = sum
        sum += 2 * func(a + (2 * i) * dx) + 4 * func(a + (2 * i + 1) * dx)
        arr.append(sum*dx/3)
        if i%2==0 and show==1:
            print("Iteration\t",i)
            if func == integral_1:
                print("Относительная Ошибка численного интегрирования методом cимпсона для первого интеграла =",abs((pi / 2 - sum * dx/3))/ (pi / 2))
            else:
                print("Относительная Ошибка численного интегрирования методом cимпсона для второго интеграла =",abs((1.29587400873 - sum*dx/3)) / (1.29587400873))
    return arr
def Graph_simpson (func_1,func_2 ,a, b, N,show=0):
    plt.title('Simpson method')
    arr_X=massiveX(N)
    arr_Y = simpson_rule(func_1,a,b,N,show)
    arr_Y2 = simpson_rule(func_2, 0, 1, N, show)
    plt.plot(arr_X,arr_Y,label='First integral' )
    plt.plot(arr_X, arr_Y2,label='Second integral')
    plt.legend()
    plt.show()
def Graph_trapezoid (func_1,func_2 ,a, b, N,show=0):
    plt.title('trapezoid method')
    arr_X=massiveX_1(N)
    arr_Y = trapezoid_rule(func_1,a,b,N,show)
    arr_Y2 = trapezoid_rule(func_2, 0, 1, N, show)
    plt.plot(arr_X,arr_Y,label='First integral' )
    plt.plot(arr_X, arr_Y2,label='Second integral')
    plt.legend()
    plt.show()
if __name__ == '__main__':
    N=10000
    left_board_1=-1
    right_board_1=1
    left_board_2 =0
    right_board_2 =1
    I_tr_1= trapezoid_rule(integral_1,left_board_1,right_board_1,nseg=N,show=0)
    I_tr_2 = trapezoid_rule(integral_2,left_board_2,right_board_2,nseg=N,show=0)
    I_sim_1 = simpson_rule(integral_1,left_board_1,right_board_1,nseg=N,show=1)
    I_sim_2 = simpson_rule(integral_2,left_board_2,right_board_2,nseg=N,show=0)
    # Graph_simpson(integral_1,integral_2,left_board_1,right_board_1,N)
    # Graph_trapezoid(integral_1,integral_2,left_board_1,right_board_1,N)
    print("#####################################")
    print("############# TOTAL ##################")
    print("Trapezoid_rule integral_1 =\t",I_tr_1[9998])
    print("Trapezoid_rule integral_2 =\t", I_tr_2[9998])
    print("Simpson_rule integral_1 =\t", I_sim_1[4998])
    print("Simpson_rule integral_2 =\t", I_sim_2[4998])
