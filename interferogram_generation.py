## Generate input dataset using Gaussian surfaces
import random
import numpy as np
import matplotlib.pyplot as plt
import math


def gaussian_surface(size=(256, 256), mean=(128, 128), variance=30.0, amplitude=10.0):
    X, Y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
    g = amplitude * np.exp(-((X - mean[0])**2 + (Y - mean[1])**2) / (2 * variance**2))
    return g


def terrain(size=(256, 256), peaks=(3, 5), x_mean=(25, 230), y_mean=(25, 230), variance=(30, 50), amplitude=(20, 80)):
    N = np.random.randint(peaks[0], peaks[1])
    h = np.zeros(size)
    for i in range(N):
        h += gaussian_surface(size, (np.random.randint(*x_mean), np.random.randint(*y_mean)), np.random.uniform(*variance), np.random.uniform(*amplitude))
    h += np.random.normal(0, 0.2, size)
    return h


def terrain2unwrap(surface=np.array(0),wavelength=0.055,baseline=100,r=500000,theta=20*math.pi/180):
    phi = 4*np.pi/wavelength*baseline/(r*np.sin(theta))*surface
    return phi


def wrap(unwrapped):
    A = 50
    B = 50
    DELTA = np.array([0,2*np.pi/3,4*np.pi/3])
    h, w = unwrapped.shape
    noise = np.random.normal(scale=np.random.uniform(0.1,140),size=(h,w,3))
    I = A + B * np.cos(unwrapped[:,:,None]-DELTA)+noise
    phi = np.arctan2(np.sum(I * np.sin(DELTA), axis=2), np.sum(I * np.cos(DELTA), axis=2))
    return phi


def get_k(unwrapped):
    k = np.round(unwrapped/(2*np.pi))
    return k



h = terrain()
PHI = terrain2unwrap(h,wavelength=0.005)
phi = wrap(PHI)
k = get_k(PHI)

plt.imshow(phi)
x, y = np.meshgrid(np.arange(len(h)),np.arange(len(h)))
fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
ax1.plot_surface(x,y,h, cmap="terrain",alpha=0.7)
plt.axis('equal')
fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
ax2.plot_surface(x,y,PHI, cmap="terrain",alpha=0.7)
fig3, ax3 = plt.subplots(subplot_kw={"projection": "3d"})
ax3.plot_surface(x,y,k, cmap="terrain",alpha=0.7)
plt.show()
plt.close('all')

