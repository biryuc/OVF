import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return math.sin(x)
def make_system(n, x_0, x_n, y_0, y_n):
    h = (x_n - x_0) / n

    a = [] #элементы под главной
    b = [] #элементы главной диагонали
    c = []  #над главной


    a.append(0)
    b.append(1)
    c.append(0)

    for i in range(1, n):
        a.append(1)
        b.append(-2)
        c.append(1)

    a.append(0)
    b.append(1)
    c.append(0)

    d = []
    x = []

    d.append(y_0)
    x.append(x_0)

    for i in range(1, n):
        x_i = x_0 + h * i
        d.append(h * h * f(x_i))
        x.append(x_i)

    d.append(y_n)
    x.append(x_n)
    A = [a, b, c]

    return A, d, x


def solve_3diag(A, d, n):
    # A = [a, b, c]

    d_new = d.copy()
    b_new = A[1].copy()
    c_new = A[2].copy()
    a_new = A[0].copy()

    for i in range(1, n + 1):
        k = a_new[i] / b_new[i - 1]
        b_new[i] -= k * c_new[i - 1]
        d_new[i] -= k * d_new[i - 1]
        print(d_new[i])
    y = [i for i in range(n + 1)]

    y[n] = d_new[n] / b_new[n]

    for i in range(n - 1, -1, -1):
        y[i] = (d_new[i] - c_new[i] * y[i + 1]) / b_new[i]

    return y

def solve_3diag_1(A, d, n):
    # A = [a, b, c]

    d_new = d.copy()
    b_new = A[1].copy()
    c_new = A[2].copy()
    a_new = A[0].copy()

    for i in range(1, n + 1):
        k = a_new[i] / b_new[i - 1]
        b_new[i] -= k * c_new[i - 1]
        d_new[i] -= k * d_new[i - 1]
        print(d_new[i])
    y = [i for i in range(n + 1)]

    y[n] = d_new[n] / b_new[n]

    for i in range(n - 1, -1, -1):
        y[i] = (d_new[i] - c_new[i] * y[i + 1]) / b_new[i]

    return d_new

def solution(x, x_0, x_n, y_0, y_n):
    c1 = (y_0 - y_n + math.sin(x_0) - math.sin(x_n)) / (x_0 - x_n)
    c2 = y_0 + math.sin(x_0) - c1 * x_0

    return -1 * math.sin(x) + c1 * x + c2

#граничные условия
y_0 = 0
y_n = 2*math.pi

x_0 = -2*math.pi            # -2*math.pi
x_n =2*math.pi             #2*math.pi
n = 20

h = (x_n - x_0)/n
A, d, x = make_system(n, x_0, x_n, y_0, y_n)
y = solve_3diag(A, d, n)
y_sol = [solution(x_i, x_0, x_n, y_0, y_n) for x_i in x]


plt.plot(x, y,  label="y(x)")
plt.plot(x, y_sol,  label="y(x) analytical")
plt.grid()
plt.xlabel('x')
plt.title('y(x)')
plt.legend()
plt.show()



#err = [abs((y[0:][i] - y_sol[0:][i])) for i in range(len(x[0:]))]
err = [abs((y[i] - y_sol[i])) for i in range(len(x[0:]))]
plt.scatter(x[0:], err,  label="|y - y_analytical|")
plt.grid()
plt.xlabel('x')
plt.ylabel('|y - y_analytical|')
plt.title('Errors')
plt.legend()
#plt.yscale("log")
plt.show()
Maxerr=[]
for i in 100,200,300,400,500,1000,10000:
    n=i
    h = (x_n - x_0) / n
    A, d, x = make_system(n, x_0, x_n, y_0, y_n)
    y = solve_3diag(A, d, n)
    y_sol = [solution(x_i, x_0, x_n, y_0, y_n) for x_i in x]
    err = [abs((y[i] - y_sol[i])) for i in range(len(x[0:]))]
    Maxerr.append(max(err))

err_d = {}
for i in range(len(x)):
    err_d[x[i]] = err[i]

#err_d

# D=solve_3diag_1(A, d, n)
# print(D)
