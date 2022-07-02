import matplotlib.pyplot as plt
import numpy as np
import cplot as cplt


# Defining logistic recursion
def logi(r, n, u0):
    u = np.zeros((n, 1))
    u[0] = u0
    for i in range(n-1):
        u[i+1] = r*u[i]*(1-u[i])
    return u

# Oscillations examples (extinction, convergence, subsequence limits, chaotic)

mu1 = 0.95
mu2 = 2.15
mu3 = 3.17
mu4 = 3.89
p1 = 30
p2 = 30
p3 = 30
p4 = 30
dep1 = .735
dep2 = .88
dep3 = .28
dep4 = .38
x1 = np.arange(0, p1, 1)
x2 = np.arange(0, p2, 1)
x3 = np.arange(0, p3, 1)
x4 = np.arange(0, p4, 1)
y1 = logi(mu1, p1, dep1)
y2 = logi(mu2, p2, dep2)
y3 = logi(mu3, p3, dep3)
y4 = logi(mu4, p4, dep4)

# Plotting examples
plt.figure(figsize=(8, 6))
plt.subplot(221)
plt.plot(x1, y1)
plt.axis([0, p1, 0, 1])
plt.title('0 ≤ µ ≤ 1\nExtinction')
plt.plot([0, p1], [0, 0], 'r-', lw=2)

plt.subplot(222)
plt.plot(x2, y2)
plt.axis([0, p2, 0, 1])
plt.title("1 ≤ µ ≤ 3\nLimite existante")
plt.plot([0, p2], [(mu2-1)/mu2, (mu2-1)/mu2], 'r-', lw=1)

plt.subplot(223)
plt.plot(x3, y3)
plt.axis([0, p3, 0, 1])
plt.title("3 ≤ µ ≤ 3.57\nValeurs d'adhérence")

plt.subplot(224)
plt.plot(x4, y4)
plt.axis([0, p4, 0, 1])
plt.title("µ ≥ 3.57\nChaos")

plt.subplots_adjust(hspace=0.5)

# Feigenbaum diagram

def logi_evo(r, n, u0):
    m = len(r)
    u = np.zeros((n, m))
    u[0] = u0
    for i in range(m):
        for k in range(n-1):
            u[k+1, i] = r[i]*u[k, i]*(1-u[k, i])
    return u[n-1]


mu = np.arange(0., 4., 0.005)
plt.figure(figsize=(9, 6))
u = np.arange(0., 1., 0.05)
nu = len(u)
for l in range(nu):
    plt.plot(mu, logi_evo(mu, 30, u[l]), 'r-', lw='0.5')
plt.title("Diagramme de Feigenbaum")


plt.show()
