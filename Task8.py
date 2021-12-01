import math
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.linalg as sla

a = 998
b = 1998
c = -999
d = -1999

def f_u(t, u, v):
    return a*u + b*v

def f_v(t, u, v):
    return c*u + d*v

t_start = 0
t_end = 1

u_start = 3
v_start = 2
N = 10

def solution(t_start, t_end, u_start, v_start, N):
    alpha = (u_start + v_start) * math.exp(t_start)
    beta = -1 * (u_start + 2 * v_start) * math.exp(1000 * t_start)


    t_s = []
    u_s = []
    v_s = []

    h = (t_end - t_start) / N


    t_s.append(t_start)
    u_s.append(u_start)
    v_s.append(v_start)

    for i in range(1, N + 1):
        t_s.append(t_start + i * h)
        u_s.append(2 * alpha * math.exp(-1 * t_s[i]) + beta * math.exp(-1000 * t_s[i]))
        v_s.append(-1 * alpha * math.exp(-1 * t_s[i]) - beta * math.exp(-1000 * t_s[i]))

    return t_s, u_s, v_s


def euler_expl( t_start, t_end, u_start, v_start, N):
    t_s = []
    u_s = []
    v_s = []

    h = (t_end - t_start) / N

    t_s.append(t_start)
    u_s.append(u_start)
    v_s.append(v_start)

    for i in range(1, N + 1):
        t_s.append(t_start + i * h)
        u_s.append(u_s[i - 1] + h * f_u(t=t_s[i - 1], u=u_s[i - 1], v=v_s[i - 1]))
        v_s.append(v_s[i - 1] + h * f_v(t=t_s[i - 1], u=u_s[i - 1], v=v_s[i - 1]))


    return t_s, u_s, v_s


def euler_impl( t_start, t_end, u_start, v_start, N):
    t_s = []
    u_s = []
    v_s = []

    h = (t_end - t_start) / N

    A = np.array([[1 - h * a, -1 * h * b], [-1 * h * c, 1 - h * d]])

    t_s.append(t_start)
    u_s.append(u_start)
    v_s.append(v_start)

    prev = np.array([u_start, v_start])

    for i in range(1, N + 1):
        t_s.append(t_start + i * h)

        cur = sla.inv(A).dot(prev)

        prev = cur

        u_s.append(cur[0])
        v_s.append(cur[1])

    return t_s, u_s, v_s

t_real,u_real,v_real=solution(t_start,t_end,u_start,v_start,N)

plt.plot(t_real,u_real,label="u-analytic")
plt.plot(t_real,v_real,label="v-analytic")

t_exp,u_exp,v_exp = euler_expl(t_start,t_end,u_start,v_start,N)

# plt.plot(t_exp,u_exp,label="u-explicit")
# plt.plot(t_exp,v_exp,label="v-explicit")


t_imp,u_imp,v_imp = euler_impl(t_start,t_end,u_start,v_start,N)
plt.plot(t_imp,u_imp,label="u-implicit")
plt.plot(t_imp,v_imp,label="v-implicit")
y=[]
# for i in range(N+1): y.append(2)
# plt.plot(t_imp,y,marker="o")

plt.legend()
plt.show()