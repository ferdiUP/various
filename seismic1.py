import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import stats as sp
from scipy import signal


# Importing Sleipner picture data, among 2 axis, 1994, 2001, 2004, 2006 processings (see Readme.md for more)
data1 = np.array([np.genfromtxt('/Users/ferdi/PycharmProjects/sismique/data/Seismic/InL94.txt', delimiter=','),
                  np.genfromtxt('/Users/ferdi/PycharmProjects/sismique/data/Seismic/InL01.txt', delimiter=','),
                  np.genfromtxt('/Users/ferdi/PycharmProjects/sismique/data/Seismic/InL04.txt', delimiter=','),
                  np.genfromtxt('/Users/ferdi/PycharmProjects/sismique/data/Seismic/InL06.txt', delimiter=',')])
data2 = np.array([np.genfromtxt('/Users/ferdi/PycharmProjects/sismique/data/Seismic/x94_2.txt', delimiter=','),
                  np.genfromtxt('/Users/ferdi/PycharmProjects/sismique/data/Seismic/x01_2.txt', delimiter=','),
                  np.genfromtxt('/Users/ferdi/PycharmProjects/sismique/data/Seismic/x04_2.txt', delimiter=','),
                  np.genfromtxt('/Users/ferdi/PycharmProjects/sismique/data/Seismic/x06_2.txt', delimiter=',')])

print(data1.shape)

l = data1.shape[0]
m = int(l/2)
# Data among the first axis
fig, ax = plt.subplots(m, m)
plt.title('Axe 1')
for k in range(m):
    ax[k, 0].imshow(data1[k, :, :], cmap='seismic')
    ax[k, 0].axis('auto')
    ax[k, 1].imshow(data1[k+2, :, :], cmap='seismic')
    ax[k, 1].axis('auto')

# Data among the second axis
fig, ax = plt.subplots(m, m)
plt.title('Axe 2')
for k in range(m):
    ax[k, 0].imshow(data2[k, :, :], cmap='seismic')
    ax[k, 0].axis('auto')
    ax[k, 1].imshow(data2[k+2, :, :], cmap='seismic')
    ax[k, 1].axis('auto')

# Computing the difference (getting remaining noise & variations)
data_res1 = np.zeros((l-1, 1001, 468))
data_res2 = np.zeros((l-1, 1001, 249))
for k in range(l-1):
    data_res1[k, :, :] = data1[k+1, :, :] - data1[k, :, :]
    data_res2[k, :, :] = data2[k+1, :, :] - data2[k, :, :]

fig, ax = plt.subplots(1, m+1)
plt.title('Différence données - Axe 1 (bruit restant & variations)')
for k in range(m+1):
    ax[k].imshow(data_res1[k, :, :], cmap='seismic')
    ax[k].axis('auto')

fig, ax = plt.subplots(1, m+1)
plt.title('Différence données - Axe 2 (bruit restant & variations)')
for k in range(m+1):
    ax[k].imshow(data_res2[k, :, :], cmap='seismic')
    ax[k].axis('auto')

noise_sample = data_res2[2, 200:1000, 0:40]
plt.figure()
plt.imshow(noise_sample, cmap='seismic')
plt.axis('auto')
plt.colorbar()
plt.title('Échantillon de bruit')

# Variance estimation (normal hypothesis)
noise = noise_sample.flatten()
print(len(noise))
plt.figure()
plt.hist(noise, 40, density=True, histtype='step')
x = np.linspace(-0.6, 0.6, 300)
m = np.mean(noise)
s = 0
n = len(noise)
for k in range(n):
    s += (noise[k] - m)**2
s2 = 1/(n-1)*s
y = 1/(np.sqrt(s2*2*np.pi))*np.exp(-x**2/(2*s2))

plt.plot(x, y)

# QQ-plot
sm.qqplot(noise, line='r', markersize='3')

# Sahpiro-Wilk test
noise_reduced = np.random.choice(noise, 4000)
shapiro_test = sp.shapiro(noise_reduced)
print(shapiro_test)

# Noise samples comparison test
noise_sample1 = data_res2[2, 600:800, 0:50]
noise_sample2 = data_res2[2, 600:800, 100:150]

plt.figure()
plt.subplot(121)
plt.imshow(noise_sample1, cmap='seismic')
plt.axis('auto')
plt.subplot(122)
plt.imshow(noise_sample2, cmap='seismic')
plt.axis('auto')

ns1 = noise_sample1.flatten()
ns2 = noise_sample2.flatten()
table = np.vstack((ns1, ns2))

test = sp.ttest_ind(ns1, ns2)
print(test)

plt.show()
