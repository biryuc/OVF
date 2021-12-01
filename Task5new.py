import math
import matplotlib.pyplot as plt
import numpy as np

def mult(t, m):
    mult = 1
    for i in range(0,m+1):
       mult = mult * (t - x[i])
    return mult

def mult_i_j(i,m):
    mult_ = 1
    for j in range(0,m):
        if j!=i:
            mult_ = mult_*(x[i] - x[j])
    return mult_

def divided_difference(m): #x,y - data, t is point to approx, m - degree of the div diff,
    k = 0
    summ = 0
    for k in range(0,m+1, 1):
        summ = summ + y[k]/mult_i_j(k,m+1)

    return summ #P(x_0, x_1, ...,x_m)

def newton_1(p):
    result = y[0]
    for k in range(1,n ):
        result = result + divided_difference(k)*mult(p, k-1)
    return result

def lagranz(t):
    z = 0
    for j in range(n+1):
        p1 = 1
        p2 = 1
        for i in range(n+1):
            if i == j:
                p1 = p1 * 1
                p2 = p2 * 1
            else:
                p1 = p1 * (float(t) - x[i])
                p2 = p2 * (x[j] - x[i])
        z = z + y[j] * p1 / p2
    return z


if __name__ == '__main__':
    n = 30
    x, y = np.zeros(n + 1), np.zeros(n + 1)
    x = [1 + k / n for k in range(0, n + 1)]
    y = [math.log(x[k]) for k in range(0, n + 1)]
    N = 100
    x_ = [min(x) + (max(x) - min(x)) / (N - 1) * i for i in range(0, N)]
    y1 = np.array([(lagranz(x_i)) for x_i in x_])
    y2 = np.array([(newton_1(x_i)) for x_i in x_])
    y_ = np.array([math.log(x_i) for x_i in x_])
    #print(y2[N-1], newton_1(2),y_[N-1],y[n], y1[N-1])
    #plt.plot(x_, (y2-y_),label = 'ErrNewton')
    #plt.yscale('log')
    plt.plot(x_, (y1-y_), label = 'ErrLagranz')
    # plt.plot(x_,y1,  label='Lagranz')
    # plt.plot(x_,y2, label='Newton')
    #plt.plot(x,y, label='log')
    plt.legend()
    plt.show()

    max_arr = []
    max_arrN = []
    N_arr = []
    for n in range(2,32):
        x, y = np.zeros(n+1), np.zeros(n+1)
        x = [1 + k/n for k in range(0,n+1) ]
        y = [ math.log(x[k]) for k in range(0, n+1)]
        #graphs
        N = 100
        x_ =[ min(x) + (max(x) - min(x))/(N-1) * i for i in range(0,N)]
        y_ = np.array([math.log(x_i) for x_i in x_])
        y1 = np.array([(lagranz(x_i)) for x_i in x_])
        err = [math.log(x_i) - lagranz(x_i) for x_i in x_ ]
        max_arr.append(max(err))
        errN = [math.log(x_i) - newton_1(x_i) for x_i in x_]
        max_arrN.append(max(errN))
        #print(max(err))
        N_arr.append(n)
   # plt.plot(N_arr, max_arrN, linewidth=0, marker='x')
    plt.plot(N_arr,max_arr, linewidth = 0 ,marker='o' )
    plt.yscale('log')
    plt.show()


