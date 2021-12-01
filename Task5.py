from math import *
import numpy as np
import matplotlib.pyplot as plt
#N = 10

def X1(k):
    arrX=[]
    arrX.append((1 + k/N))
    return arrX
def Y1(k):
    arrY=[]
    arrY.append(log(X(k,N)))
    return arrY
def X(k,N):
    return (1 + k/N)
def Y(k,N):
    return log(X(k,N))
def Graphic(N):
    Y_=[]
    X_=[]
    for k in range(N):
        Y_.append(Y1(k))
        X_.append(X1(k))

    print(X_)
    print(Y_)
    plt.plot(X_, Y_, label='Polynom')
    plt.legend()
    plt.show()


def l( i, x,N):
    l = 1
    for k in range(N):
        if (k != i):
            l *= (x - X(k,N))
    return l

def LagrangePolynomial(x,N):
    P = 0
    result=[]
    for i in range(N):
        P += Y(i,N) * l(i, x,N) / l(i, X(i,N),N)
        result.append(P)
    return P


def NewtonPolynomial(x,N):
    a =np.zeros(N+1)
    for i in range(N):
        a[i] = Y(i,N)
        for j in range(i):
            a[i] = (a[i] - a[j]) / (X(i,N) - X(j,N))
    res = a[N]
    i=N-1
    while i>=0:
        res = res * (x - X(i,N)) + a[i]
        i-=1
    return res

if __name__ == '__main__':
    samples = 1000
    a = 1.9999999
    b = 2
    delta = (b - a) / samples
    Lresult=[]
    Nresult=[]
    Lresult0 = []
    Nresult0 = []
    errLresult = []
    errNresult = []
    arrX=[]
    resL_old=0
    resN_old=0
    # for j in range(N):
    #     errLresult.append(resL_old)
    #     errNresult.append(resN_old)
    #     for i in range(samples):
    #         x = a + delta * i
    #         arrX.append(x)
    #         resL = LagrangePolynomial(x,N) - log(x)
    #         resN = NewtonPolynomial(x,N) - log(x)
    #         if resL_old <= resL:
    #             resL_old=resL
    #         if resN_old <= resN:
    #             resN_old=resN
    N=40
    print(NewtonPolynomial(2,N),log(2))
    for i in range(samples):
         x = a + delta * i
         arrX.append(x)
         resL0 = LagrangePolynomial(x, N)
         resN0 = NewtonPolynomial(x, N)
         Lresult0.append(resL0)
         Nresult0.append(resN0)
         resL=LagrangePolynomial(x,N) - log(x)
         resN=NewtonPolynomial(x,N) - log(x)
         Lresult.append(resL)
         Nresult.append(resN)
    maxL=max(Lresult)
    maxN = max(Nresult)
    print(maxN,maxL)
    errL=[7.769693410875078e-07,3.611499987954403e-11,7.879023367074467e-08,9.878136198748777e-05,0.010136244335585354]
    errN=[7.769693348702589e-07,1.2240430891097276e-11,1.830627871513002e-09,1.3648824292999961e-09,8.892885242417492e-06]
    arr=[10,20,30,40,50]
    Y_ = []
    X_ = []
    for k in range(N):
        Y_.append(Y1(k))
        X_.append(X1(k))
    plt.plot(arrX,Lresult0,label='Lagrange')
    plt.plot(arrX,Nresult0,label='Newton')
    # plt.plot(X_, Y_, marker='x', label='Polynom')
    # plt.plot(X_, Y_, marker = 'x', label='Polynom')
    plt.legend()
    plt.show()
    plt.plot(arr,errL,label='Lagrange')
    #plt.plot(arr,errN,label='Newton')
    plt.legend()
    plt.show()
    #Graphic()
    # for i in range(len(arrX)):
    #     print("arrX = \t",arrX[i],"Lresult = \t",Lresult[i],"Nresult = \t",Nresult[i])
    # for i in range(4,16):
    #     n=i
    #     x = np.array([1 + k / n for k in range(n)])
    #     y = np.array([np.log(x_i) for x_i in x])
    #     pointNumber = 100
    #     xnew = np.linspace(np.min(x), np.max(x), pointNumber)
    #     ynew = LagrangePolynomial(x, y, pointNumber)
    #     err = ynew - np.log(xnew)
# график P и узлы на интервале от 1 до 2