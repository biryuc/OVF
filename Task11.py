import math
import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt




def psi0_analytical(x):
    return (1/math.pi)**(1/4) * math.exp(-0.5*(x**2))

E0_sol = 1/2
x_start = -10
x_end = 10
N = 100

h = (x_end - x_start) / N

x_s_sol = []
psi0_sol = []

for i in range(N):
    x_s_sol.append(x_start + i * h)
    psi0_sol.append(psi0_analytical(x_s_sol[i]))

# нормировать на 1
sol_norm = sla.norm(psi0_sol)
psi0_sol = [psi0_sol[i] / sol_norm for i in range(len(psi0_sol))]



# plt.figure(figsize=(7,7))
# plt.plot(x_s_sol, psi0_sol, 'm+', label="psi_0(x)")
# plt.grid()
# plt.title('Волновая функция основного состояния, аналитическое решение')
# plt.xlabel('x')
# plt.ylabel('psi0(x)')
# plt.legend()
# plt.show()

def U(x):
    return (x**2)/2


def make_matrix(N, x_start, x_end):
    h = (x_end - x_start) / N

    a = []
    b = []
    c = []

    for i in range(0, N):
        xi = x_start + i * h
        a.append(-0.5 / (h ** 2))
        b.append(1 / (h ** 2) + U(xi))
        c.append(-0.5 / (h ** 2))

    a[0] = 0
    c[N - 1] = 0

    A = [a, b, c]

    return A


def solve_3diag(A, d, N):
    # A = [a, b, c]

    d_new = d.copy()
    b_new = A[1].copy()
    c_new = A[2].copy()
    a_new = A[0].copy()

    for i in range(0, N):
        k = a_new[i] / b_new[i - 1]
        b_new[i] -= k * c_new[i - 1]
        d_new[i] -= k * d_new[i - 1]
        #print(d_new[i])
    y = [i for i in range(N)]

    y[N - 1] = d_new[N - 1] / b_new[N - 1]

    for i in range(N - 2, -1, -1):
        y[i] = (d_new[i] - c_new[i] * y[i + 1]) / b_new[i]

    return y


x_s = [(x_start + i*h) for i in range(N)]


H = make_matrix(N, x_start, x_end)  # трехдиагональная матрица
def vector():
    psi_next_ = [1 for i in range(N)]
    psi_next_[0]=0
    psi_next_[N-1] = 0
    return psi_next_
psi_next = vector()
#psi_next = [i/N for i in range(N)]
print(len(psi_next))# как начальный случайный вектор возьму просто N точек от 0 до 1
psi_prev = psi_next

K=1
for i in range(1,10):
    for k in range(0, K):
        psi_prev = psi_next
        psi_next = solve_3diag(H, psi_prev, N)

    E0 = sla.norm(psi_prev) / sla.norm(psi_next)
    print("K=",K,"\t","errE_0=",abs((E0_sol-E0)/E0_sol))
    K+=1


psi0_norm = sla.norm(psi_next)
psi0 = [psi_next[i] / psi0_norm for i in range(len(psi_next))]

plt.figure(figsize=(7,7))
plt.plot(x_s, psi0, 'g+', label="Численное решение")
plt.plot(x_s, psi0_sol, 'y', label="Аналитическое решение")
plt.grid()
plt.title('Волновая функция основного состояния')
plt.xlabel('x')
plt.ylabel('psi0(x)')
#plt.yscale('log')
plt.legend()
plt.show()

print("Вычисленное E_0 = \t",E0)
print("E0_analytic -  E_0 = \t",abs((E0_sol - E0)/E0_sol))
err = [abs(psi0[i] - psi0_sol[i]) for i in range(len(x_s))]



plt.figure(figsize=(7,7))
plt.plot(x_s, err, 'b+', label="|psi0(x) - psi0_sol(x)|")
plt.grid()
plt.title('Абсолютная ошибка:')
plt.xlabel('x')
plt.ylabel('|psi0(x) - psi0_sol(x)|')
#plt.yscale('log')
plt.legend()
plt.show()