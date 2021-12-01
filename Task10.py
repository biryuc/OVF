import numpy as np
import matplotlib.pyplot as plt

L = 1


def make_matrix(u_last, tau, h, n):    # returns matrix and rhs for finding u(x, t + tau)
    # right-hand side
    d = np.zeros(n + 1)
    for j in range(1, n):
        d[j] = u_last[j] + tau / 2 / (h**2) * (u_last[j + 1] - 2 * u_last[j] + u_last[j - 1])
    # mat
    mat = np.zeros([n + 1, n + 1])
    mat[0, 0] = 1
    mat[n, n] = 1
    for i in range(1, n):
        mat[i, i + 1] = -tau / 2 / (h**2)
        mat[i, i] = 1 + tau / (h**2)
        mat[i, i - 1] = -tau / 2 / (h**2)
    return mat, d


def solve_3diag(mat, d):     # returns u(x, t + tau)
    num = len(d)
    for i in range(1, num):
        ksi = mat[i, i - 1] / mat[i - 1, i - 1]
        mat[i, i - 1] = 0
        mat[i, i] = mat[i, i] - ksi * mat[i - 1, i]
        d[i] = d[i] - ksi * d[i - 1]
    y = np.zeros(num)
    y[num - 1] = d[num - 1] / mat[num - 1, num - 1]
    for j in range(num - 2, -1, -1):
        y[j] = (d[j] - mat[j, j + 1] * y[j + 1]) / mat[j, j]
    return y


def solution_and_temp(n, m, tau):
    h = L / n
    x = np.array([i * h for i in range(n + 1)])
    u = np.array([[i * (1 - i / L) ** 2 for i in x]])
    temp_max = np.array([])
    time = np.array([])
    for i in range(m):
        time = np.append(time, i * tau)
        temp_max = np.append(temp_max, u[-1, :].max())
        mat, f = make_matrix(u[-1, :], tau, h, n)
        u = np.append(u, [solve_3diag(mat, f)], axis=0)

    return u, x, temp_max, time


n1 = 200  # шаг для h
m1 = 500 # количество итераций для T
tau1 = 0.02
u1, x1, T_max, t1 = solution_and_temp(n1, m1, tau1)

plt.plot(t1,T_max,"m+",label="Tmax(t)")
plt.legend()
plt.show()
