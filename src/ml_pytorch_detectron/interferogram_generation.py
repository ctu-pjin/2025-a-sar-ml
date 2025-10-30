## Generate input dataset using Gaussian surfaces
import random
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.transform import resize
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gaussian_surface(size=(256,256), mean=(128,128), variance=30.0, amplitude=10.0, negative=False):
    print(variance,amplitude)
    X, Y = np.meshgrid(np.arange(size[0]),np.arange(size[1]))
    g = amplitude*np.exp(-((X-mean[0])**2+(Y-mean[1])**2)/(2*variance**2))
    if negative:
        g = -g
    return g


def gaussian_surface_faster(size=(256,256), mean=(128,128), variance=30.0, amplitude=10.0, negative=False):
    r = int(3*variance)
    x0, y0 = mean
    x_min, x_max = max(0,x0-r), min(size[0],x0+r)
    y_min, y_max = max(0,y0-r), min(size[1],y0+r)
    X, Y = np.meshgrid(np.arange(x_min,x_max), np.arange(y_min,y_max))
    g_area = amplitude*np.exp(-((X-x0)**2+(Y-y0)**2)/(2*variance**2))
    if negative:
        g_area = -g_area
    g = np.zeros(size)
    g[x_min:x_max,y_min:y_max] = np.transpose(g_area)
    return g


def incline_plane(image_size=(256, 256), plane_size=(20, 20), origin_position=(0, 0), x_direction=0., y_direction=0., intercept=0.):
    assert origin_position[0] + plane_size[0] <= image_size[0] and origin_position[1] + plane_size[1] <= image_size[1], "Plain is not completely within given image size."
    p = np.zeros(image_size)
    x = np.arange(plane_size[0])
    y = np.arange(plane_size[1])
    X, Y = np.meshgrid(x, y)
    z = np.tan(x_direction) * X + np.tan(y_direction) * Y + intercept
    p[origin_position[0]:origin_position[0] + plane_size[0], origin_position[1]:origin_position[1] + plane_size[1]] = np.transpose(z)
    return p


def terrain(size=(256,256),peaks=(3,5),x_mean=(25,230),y_mean=(25,230),variance=(30,50),plane=False):
    print(peaks)
    N = np.random.randint(peaks[0], peaks[1])
    print(N)
    h = np.zeros(size)
    L = max(size)
    for i in range(N):
        A_base = 0.01*L
        sigma_ref = 0.05*L
        sigma = np.random.uniform(*variance)
        A = A_base*np.sqrt(sigma_ref/sigma)
        A = np.clip(A,0.2*A_base,0.02*L)
        h += gaussian_surface_faster(size, (np.random.randint(*x_mean),np.random.randint(*y_mean)),sigma,A,random.getrandbits(1))
    h += np.random.normal(0,0.02*(0.01*L),size)
    if plane:
        x_size = np.random.randint(10,75)
        y_size = np.random.randint(10,75)
        x_origin = np.random.randint(0,size[0]-x_size)
        y_origin = np.random.randint(0,size[1]-y_size)
        p = incline_plane(image_size=size, plane_size=(x_size,y_size), origin_position=(x_origin,y_origin),x_direction=np.random.uniform(-np.pi/6,np.pi/6),y_direction=np.random.uniform(-np.pi/6,np.pi/6),intercept=np.random.uniform(np.min(h),np.max(h)))
        h = np.where(p!=0,p,h)
    if np.min(h) < 0:
        h = h-np.min(h)
    return h


def terrain2unwrap(surface=np.array(0), wavelength=0.055, baseline=100, r=500000, theta=20 * math.pi / 180, image_size=(256, 256), dem_resolution=1):
    new_shape = image_size
    surface_ds = resize(surface, new_shape, anti_aliasing=True)
    phi = 4 * np.pi / wavelength * baseline / (r * np.sin(theta)) * surface_ds
    return phi


def wrap(unwrapped):
    A = 50
    B = 50
    DELTA = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
    h, w = unwrapped.shape
    n = np.random.normal(scale=np.random.uniform(0.1, 140), size=(h, w, 3))
    I = A + B * np.cos(unwrapped[:, :, None] - DELTA) + n
    phi = np.arctan2(np.sum(I * np.sin(DELTA), axis=2), np.sum(I * np.cos(DELTA), axis=2))
    return phi


def get_k(unwrapped):
    k = np.round(unwrapped / (2 * np.pi))
    return k


def get_terrain_params(area=(5000, 5000)):
    L = max(area)
    A_area = area[0]*area[1]
    d = 2/1e6
    N = max(1,int(d*A_area))
    peaks_range = (int(0.8*N),int(1.2*N))
    variance_range = (0.005*L,0.2*L)
    amplitude_range = (0.002*L,0.02*L)
    return peaks_range, amplitude_range, variance_range


def generate_set(area=(5000,5000), image_sizes=[(256,256)], set_size=1):
    peaks, amplitude, variance = get_terrain_params(area)
    L = max(area)
    h = np.zeros([set_size, 1, *area])
    for i in range(set_size):
        h[i,0,:,:] = terrain(size=area, peaks=peaks, x_mean=(0,area[0]), y_mean=(0,area[1]),variance=variance, plane=False)
    all_results = []
    for image_size in image_sizes:
        PHI = np.zeros([set_size, 1, *image_size])
        phi = np.zeros([set_size, 1, *image_size])
        k = np.zeros([set_size, 1, *image_size])
        for i in range(set_size):
            PHI[i, 0, :, :] = terrain2unwrap(surface=h[i, 0, :, :], wavelength=0.055, baseline=100, r=800000, theta=30*math.pi/180, image_size=image_size, dem_resolution=1)
            phi[i, 0, :, :] = wrap(unwrapped=PHI[i, 0, :, :])
            k[i, 0, :, :] = get_k(unwrapped=PHI[i, 0, :, :])
        all_results.append((torch.tensor(PHI, device=device),
                            torch.tensor(phi, device=device),
                            torch.tensor(k, device=device)))
    h = torch.tensor(h, device=device)
    return h, all_results









h, results = generate_set(area=(5000,5000),image_sizes=[(256,256)],set_size=1,)
phi = results[0][1]

plt.imshow(h[0,0,:,:].cpu().numpy())
plt.show()
plt.imshow(phi[0,0,:,:].cpu().numpy(),cmap='gray')
plt.show()
plt.imshow(phi[0,0,:,:].cpu().numpy())
x_1, y_1 = np.meshgrid(np.arange(len(h[0,0,:,:])),np.arange(len(h[0,0,:,:])))
fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
ax1.plot_surface(x_1,y_1,h[0,0,:,:].cpu().numpy(), cmap="terrain",alpha=0.7)
plt.axis('equal')
plt.show()

plt.close('all')


