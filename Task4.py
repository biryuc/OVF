from math import *
import numpy as np
import matplotlib.pyplot as plt

def Bessel (t,x,m):
    return np.cos(m*t - x*np.sin(t)) / np.pi

def derivative(func, var):   # сумма левой и правой конечных разностей
    h = var[1] - var[0]
    print("H=",h)
    der = []
    for i in range(1, len(func)-1):
        der.append((func[i + 1] - func[i-1]) / h / 2)
    return der

def simpson_rule(func, a, b, nseg,show,x,m):
    if nseg%2 == 1:
        nseg += 1
    dx = 1.0 * (b - a) / nseg
    arr=[]
    sum = (func(a,x,m) + 4 * func(a + dx,x,m) + func(b,x,m))
    for i in range(1, int(nseg*0.5)):
        last_sum = sum
        sum += 2 * func(a + (2 * i) * dx,x,m) + 4 * func(a + (2 * i + 1) * dx,x,m)
        arr.append(sum*dx/3)
        if i%2==0 and show==1:
            print("Iteration\t",i)
            print("Ошибка численного интегрирования методом трапеций = ", (sum - last_sum) * dx)
    return sum*dx/3
if __name__ == '__main__':
    num = 17000 # 1/170000 ** 2 ~ E-10
    num_tau = 16  # ~E-6
    x_val = np.linspace(0, 2 * np.pi, num + 1)
    arrX = np.linspace(0,num+1,num+1)
    a=0
    b=pi
    tau = np.linspace(0, np.pi, num_tau + 1)
    precision = 1e-10
    B_1=[]
    B_0=[]
    SUMM=np.zeros(num+1)
    for k in x_val:
        B_1.append(simpson_rule(Bessel, a, b, num_tau,show=0,x=k,m=1))
    x_val = np.insert(x_val, 0, x_val[0] - x_val[1])
    x_val = np.append(x_val, x_val[-1] + x_val[2])
    for k in x_val:
        B_0.append(simpson_rule(Bessel, a, b, num_tau,show=0,x=k,m=0))
    #plt.plot(x_val, B_0, label='B0')
    B0_der = derivative(B_0, x_val)
    for k in range(num+1):
        SUMM[k]=B_1[k]+B0_der[k]
    plt.plot(arrX,SUMM,label='B0_der+B1')
    # print(sum((np.array(B_1) + np.array(B0_der))))
    # print(len(B_1),len(B0_der))
    x_val = np.linspace(0, 2 * np.pi, num + 1)
    # plt.plot(x_val, B_1, label='B1')
    # plt.plot(x_val, B0_der, label='B0_der')
    plt.legend()
    plt.show()
