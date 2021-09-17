from math import *

import numpy as np
import matplotlib.pyplot as plt
U0=1
A=1
plankConst=1#  6.62607015*pow(10, -34)
def massiveX():
    x = np.zeros((100))
    for i in range(1,100):
        x[i]=i*0.01
    return x

def equation_solution(E):
    ksi = E/U0
    constant = 1 # constant = 2 * M * pow(A, 2) * U0 / pow(plankConst, 2)
    return cos(sqrt(constant * (1 - ksi)))/sin(sqrt(constant * (1 - ksi))) - sqrt(1 /ksi - 1)


def dichotomyMethod(left_board,right_board):
    delta0 = 2
    delta = pow(10, -6)
    leftboardResult = equation_solution(left_board)
    rightboardResult = equation_solution(right_board)
    N=(log(delta0 / delta) / log(2))
    for i in range(int(N+1)):
        mid = (left_board + right_board)/2
        f_mid = equation_solution(mid)
        if (abs(f_mid) < delta):
            return mid
        if (f_mid * rightboardResult < 0):
            left_board = mid
        elif (f_mid*leftboardResult<0):
            right_board = mid
    return (left_board + right_board)/2

def Graphic_1():
    plt.title('Graphics ctg()=sqrt()')
    plt.ylabel('function value')
    array= massiveX()
    plt.plot(array,[equation_solution(i) for i in array])
    plt.show()
#𝜙(𝑥) := 𝑓(𝑥)+x
def fi(x, Lambda):
    fi= x - Lambda*((cos(sqrt(A*(1-x)))/sin(sqrt(A*(1-x))) - sqrt(1/x - 1) ))
    return fi

def diffFUNCTION(E):
    func = 1/(sin(sqrt(A*(1-E))))**2 * 1/2 *A/sqrt(A*(1-E)) + 1/2*(1/E**2)*(1/sqrt(1/E - 1)) + 1
    return func

def simpleiterationMethod():
    delta = pow(10, -6)
    x = 0.5
    Lambda = 1/diffFUNCTION(x)*0.999

    x_old= 0
    i = 0
    arrayX =[]
    while abs(x - x_old) >= delta:
        i+=1
        x_old = x
        if (x_old > 0) and x_old < 1:
            x = fi(x_old, Lambda)
            arrayX += [x]
        else:
            print('out of boundaries')
            break
    print("simpleiterationMethod x=",x,"iteration counter N = ", i)
    return x

def evaluate():
    delta = pow(10, -6)
    delta0= 2
    Lambda = 1 / diffFUNCTION(0.4) * 0.999
    N=log(delta/delta0)/(-log(abs(diffFUNCTION(fi(0.4,Lambda)))))
    return N
N=evaluate()
print(N)
def Graphic_2():
    plt.title('simpleiterationMethod')
    plt.ylabel('function value')
    array = simpleiterationMethod()
    Lambda = 1/diffFUNCTION(0.4) * 0.999
    x=massiveX()
    plt.plot(x, [fi(i, Lambda) for i in x])
    plt.plot(array, [fi(i,Lambda) for i in array])
    plt.show()

def NewtonMethod():
    delta = pow(10, -6)
    x = 0.001
    x_old = 0
    i = 0
    Lambda = 1/diffFUNCTION(x)
    arrayX=[]
    while abs(x - x_old) >= delta:
        i += 1
        x_old = x
        if (x_old > 0) and x_old < 1:
            x = fi(x_old, Lambda)
            arrayX += [x]
        else:
            print('out of boundaries')
            break
    print("NewtonMethod x=", x, "iteration counter N = ", i)
    return arrayX
def Graphic_3():
    plt.title('NewtonMethod')
    plt.ylabel('function value')
    array= NewtonMethod()
    Lambda = 1 / diffFUNCTION(0.4) * 0.1
    x= massiveX()
    plt.plot(x, [fi(i, Lambda) for i in x])
    plt.plot(array, [fi(i,Lambda) for i in array])
    plt.show()

dichotomyMethodResult=dichotomyMethod(0.0001,0.9999)
SimpleiterationMethod=simpleiterationMethod()
print("dichotomyMethod C=",dichotomyMethodResult)
# Graphic_1()
# Graphic_2()
Graphic_3()
