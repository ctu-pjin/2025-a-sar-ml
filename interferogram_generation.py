## Generate input dataset using Gaussian surfaces
import random
import numpy as np
import matplotlib.pyplot as plt
import math

def gaussian_surface(size=None, mean=None, variance=1, amplitude=10):
    if size is None:
        size = [256,256]
    if mean is None:
        mean = [size[0]/2, size[1]/2]
    g = np.zeros(size)
    for x in range(size[0]):
        for y in range(size[0]):
            g[x,y] = amplitude*math.exp(-((x-mean[0])**2+(y-mean[1])**2)/(2*variance**2))

    return g

# print(gaussian_surface(variance=50, amplitude=10))
# plt.imshow(gaussian_surface(variance=50, amplitude=10))
# plt.show()




