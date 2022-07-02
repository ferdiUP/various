import math as m
import matplotlib.pyplot as plt
import random as rd
from mpmath import li


# Defining primality test
def is_prime(n):
    if n == 1:
        return False
    if n % 2 == 0 and n > 2:
        return False
    return all(n % i for i in range(3, int(m.sqrt(n)) + 1, 2))


def f_test(n, r):
    X = []
    for l in range(r):
        X = X + [rd.randrange(2, n, 1)]
    if any(a**(n-1) % n != 1 for a in X):
        return False
    else:
        return True


def experience1(n, r):
    X = []
    res = []
    for l in range(r):
        X = X + [rd.randrange(2, n, 1)]
        if X[l] ** (n - 1) % n != 1:
            res += [X[l]]
    return res


# Plotting pi, x/ln x and li functions
def experience2(n):
    # Création d'une subdivision régulière de [2, n]
    N = 3*n
    x = [2]
    y1 = [li(2)]
    y2 = [2/m.log(2, m.e)]
    pi = [0] * n
    for i in range(3, n):
        pi[i] = pi[i-1]
        if is_prime(i) == True:
            pi[i] += 1
    for k in range(1, N):
        x += [x[k-1] + n/N]
        y1 += [li(x[k])]
        y2 += [x[k]/m.log(x[k], m.e)]
    plt.step(range(n), pi, label="pi(x)")
    plt.plot(x, y1, label="li(x)")
    plt.plot(x, y2, label="x/ln(x)")
    plt.legend()
    plt.show()


def g_test(n):
    for k in range(n):
        if is_prime(k) == True:
            if is_prime(n-k) == True:
                return (True, k, n-k)


def experience3(n):
    X = []
    for k in range(4, n, 2):
        if g_test(k)[0] == False:
            X += [k]
    return X


# Tests
print(is_prime(29))
n = 50000
r = 10
print(f_test(n, r))
print(experience1(n, r))
experience2(10000)
print(g_test(33))
print(experience3(50000))
