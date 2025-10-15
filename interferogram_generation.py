## Generate input dataset using Gaussian surfaces
import random
import numpy as np
import matplotlib.pyplot as plt
import math


def gaussian_surface(size=(256,256), mean=(0, 0), variance=1.0, amplitude=10.0):
    g = np.zeros(size)
    for x in range(size[0]):
        for y in range(size[0]):
            g[x,y] = amplitude*math.exp(-((x-mean[0])**2+(y-mean[1])**2)/(2*variance**2))
    return g


def terrain(size=(256,256), peaks=(5,5), x_mean=(25,230), y_mean=(25,230), variance=(30,50), amplitude=(20,80)):
    N = random.randint(peaks[0], peaks[1])
    print(N)
    h = np.zeros(size)
    for i in range(N):
        h += gaussian_surface(size, (random.randint(x_mean[0],x_mean[1]),random.randint(y_mean[0],y_mean[1])),random.uniform(variance[0],variance[1]),random.uniform(amplitude[0],amplitude[1]))
    return h




h = terrain(peaks=(3,5))
print(h)
plt.imshow(h)

x, y = np.meshgrid(np.arange(len(h)),np.arange(len(h)))
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(x,y,h, cmap="terrain",alpha=0.7)
plt.axis('equal')

plt.show()




