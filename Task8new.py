import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.linalg as sla
import time
#u' = 998u+1998v = f(u,v)
#v' = -999u - 1999v = g(u,v)
a = 998
b = 1998
c = -999
d = -1999
def f(u_i, v_i):
    return 998*u_i+1998*v_i
def g(u_i, v_i):
    return -999*u_i-1999*v_i

def explicit(n, h, u_0, v_0):
    u = np.zeros(n)
    u[0] = u_0
    v = np.zeros(n)
    v[0] = v_0
    for i in range(n-1):
        u[i + 1] = u[i] + h * f(u[i], v[i])
        v[i + 1] = v[i] + h * g(u[i], v[i])
    return u, v

def simple_iterations(un, vn,  h, N):# хотим приблизить n+1 й номер u  и v , подаем nй номер, N число итераций
    u_0 = un
    u_1 = un+1
    v_0 = vn
    v_1 = vn+1
    iteration = 0
    gv,gu,fv,fu = -999,-1999, 998,1998
    #while (abs(u_1 - u_0) >= d and  abs(v_1 - v_0) >= d): #u_1 это u_s+1, а u_0 это u_s
    while iteration < N:
        iteration += 1
        u_0 = u_1
        u_1 = u_0 + (1 - h*gv)*(un - u_0 + h*f(u_0, v_0)) + h*fv*(vn - v_0 + h* g(u_0, v_0))
        v_1 = v_0 + h*gu*(un - u_0 + h*f(u_0, v_0)) + (1-h*fu)*(vn - v_0 + h* g(u_0, v_0))
    return u_1, v_1

def implicit(n, h,  u_0, v_0, num_it):# неявный
    u = np.zeros(n)
    u[0] = u_0
    v = np.zeros(n)
    v[0] = v_0
    for i in range(n - 1):
        u_1, v_1 = simple_iterations(u[i], v[i],  h, N=num_it)
        u[i + 1] = u[i] + h * f(u_1, v_1)
        v[i + 1] = v[i] + h * g(u_1, v_1)
    return u, v

t_start = 0
t_end = 1

def euler_impl( u_start, v_start, h,N):
    u_s = []
    v_s = []

    A = np.array([[1 - h * a, -1 * h * b], [-1 * h * c, 1 - h * d]])
    u_s.append(u_start)
    v_s.append(v_start)

    prev = np.array([u_start, v_start])

    for i in range(1, N):
        cur = sla.inv(A).dot(prev)
        prev = cur

        u_s.append(cur[0])
        v_s.append(cur[1])


    return u_s, v_s




n = 10000
h = 0.001

t_arr = np.array([ h*i for i in range(n)])
#t_arr = np.array([ i/n for i in range(n)])
u_0, v_0 =  3, 2
#solution
c1 = u_0 + v_0
c2 = -u_0 - 2 * v_0
real_u = np.array([c1 * 2 * np.exp(-t_arr[i]) + c2 * np.exp(-1000 *t_arr[i]) for i in range(n)])
real_v = np.array([-c1 * np.exp(-t_arr[i]) - c2 * np.exp(-1000 * t_arr[i]) for i in range(n)])
#implicit
u_imp, v_imp = implicit(n, h,  u_0, v_0, num_it = 1 )
err1_imp = np.array([abs(u_imp[i] - real_u[i]) for i in range(n)])

#excplicit
u_exp, v_exp = explicit(n, h, u_0, v_0 )
# err1_exp = np.array([abs(u_exp[i] - real_u[i]) for i in range(n)])
# plt.plot(t_arr, err1_exp,label = 'err1_exp')
# plt.plot(t_arr, err1_imp,label = 'err1_imp')
# plt.legend()
# plt.show()
#graohs
u_imp,v_imp = euler_impl(u_0,v_0,h,n)
plt.plot(t_arr, real_u,label = 'real_u')
plt.plot(t_arr, real_v,label = 'real_v')
# plt.plot(t_arr, u_imp, label = 'u_imp')
# plt.plot(t_arr, v_imp, label = 'v_imp')
# plt.plot(t_arr, u_exp, label = 'u_exp')
# plt.plot(t_arr, v_exp, label = 'v_exp')
plt.legend()
plt.show()



#checking accuracy IMPLICIT
# h = h/2
# t_arr2 = np.array([ h*i for i in range(2*n)])
# u2, v2 = implicit(2*n, h, u_0, v_0, num_it = 2 )
# real_u2 = np.array([c1 * 2 * np.exp(-t_arr2[i]) + c2 * np.exp(-1000 *t_arr2[i]) for i in range(2*n)])
# real_v2 = np.array([-c1 * np.exp(- t_arr2[i]) - c2 * np.exp(-1000 * t_arr2[i]) for i in range(2*n)])
# err2_impl = np.array([abs(u2[i] - real_u2[i]) for i in range(2*n)])
# p = math.log((max(err1_imp)/max(err2_impl)),2)
# print('порядок точности неявного метода:',p)

#excplicit
# n = 10000
# h = 0.00001
# t_arr = np.array([ h*i for i in range(n)])
# u, v = explicit(n, h, u_0, v_0 )
# err1 = np.array([abs(u[i] - real_u[i]) for i in range(n)])











#checking accuracy
h = h/2
# t_arr2 = np.array([ h*i for i in range(2*n)])
# u2, v2 = explicit(2*n, h, u_0, v_0,)
# err2 = np.array([abs(u2[i] - real_u2[i]) for i in range(2*n)])
# p = math.log((max(err1)/max(err2)),2)
# print('порядок точности явного метода:', p)
