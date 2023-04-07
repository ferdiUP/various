import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# Définir les équations différentielles
def eq1(state, t):
    x, y, z = state
    return np.array([10 * (y - x), x * (28 - z) - y, x * y - (8 / 3) * z])


def eq2(state, t):
    x, y, z = state
    return np.array([10 * (y - x), x * (28 - z) - y + 10, x * y - (8 / 3) * z])


# Conditions initiales
state1 = [1, 1, 1]
state2 = [1, 1, 1]

# Résoudre les équations différentielles pour obtenir les trajectoires des attracteurs
t = np.arange(0, 50, 0.01)
trajectory1 = odeint(eq1, state1, t)
trajectory2 = odeint(eq2, state2, t)

# Plotting des attracteurs étranges
fig = plt.figure(figsize=(12, 6))

# Attracteur 1
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(trajectory1[:, 0], trajectory1[:, 1], trajectory1[:, 2], lw=0.5)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Attracteur 1')

# Attracteur 2
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot3D(trajectory2[:, 0], trajectory2[:, 1], trajectory2[:, 2], lw=0.5)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Attracteur 2')

plt.show()
