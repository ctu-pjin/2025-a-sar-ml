## Generate input dataset using Gaussian surfaces
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def gaussian_surface_faster(size=(256,256), pixel_size=(1.0,1.0), mean=(128,128), sigma=10.0, amplitude=10.0, negative=False):
    x0, y0 = mean
    pixel_size_x, pixel_size_y = pixel_size
    r_x = max(1, int(3 * sigma / pixel_size[0]))
    r_y = max(1, int(3 * sigma / pixel_size[1]))
    x_min = max(0, int(mean[0] - r_x))
    x_max = min(size[1], int(mean[0] + r_x + 1))
    y_min = max(0, int(mean[1] - r_y))
    y_max = min(size[0], int(mean[1] + r_y + 1))
    if x_max <= x_min or y_max <= y_min:
        g = np.zeros(size)
        return g
    xv = np.arange(x_min, x_max) * pixel_size_x
    yv = np.arange(y_min, y_max) * pixel_size_y
    X, Y = np.meshgrid(xv, yv)
    X0 = x0 * pixel_size_x
    Y0 = y0 * pixel_size_y
    g_area = amplitude * np.exp(-(((X - X0)**2 + (Y - Y0)**2) / (2 * sigma**2)))
    if negative:
        g_area = -g_area
    g = np.zeros(size)
    g[y_min:y_max, x_min:x_max] = g_area
    return g

def incline_plane(image_size=(256,256), pixel_size=(1.0,1.0), plane_size=(20,20), origin_position=(0,0), x_direction=0., y_direction=0., intercept=0.):
    assert origin_position[0] + plane_size[0] <= image_size[0] and origin_position[1] + plane_size[1] <= image_size[1], "Plain is not completely within given image size."
    p = np.zeros(image_size)
    x = np.arange(plane_size[0]) * pixel_size[0]
    y = np.arange(plane_size[1]) * pixel_size[1]
    X, Y = np.meshgrid(x, y)
    z = np.tan(x_direction) * X + np.tan(y_direction) * Y + intercept
    p[origin_position[1]:origin_position[1] + plane_size[1], origin_position[0]:origin_position[0] + plane_size[0]] = z
    return p

def terrain(size=(256,256), pixel_size=(1.0,1.0), peaks=(3,5), variance_scale=(3,50), amplitude_scale=1.0, plane=False):
    N = np.random.randint(peaks[0], peaks[1])
    h = np.zeros(size)
    for i in range(N):
        A = np.random.exponential(scale=1.0) * amplitude_scale
        print(amplitude_scale/A)
        sigma = np.random.exponential(scale=5*amplitude_scale/A)*variance_scale
        print(sigma)
        x0 = np.random.randint(0, size[1])
        y0 = np.random.randint(0, size[0])
        h += gaussian_surface_faster(size, pixel_size, (x0,y0), sigma, A, random.getrandbits(1))
    base_noise = 0.03 * max(1.0, amplitude_scale)
    h += np.random.normal(0, base_noise, size)
    if plane:
        x_size = np.random.randint(10,75)
        y_size = np.random.randint(10,75)
        x_origin = np.random.randint(0,size[1]-x_size)
        y_origin = np.random.randint(0,size[0]-y_size)
        p = incline_plane(image_size=size, pixel_size=pixel_size, plane_size=(x_size,y_size), origin_position=(x_origin,y_origin), x_direction=np.random.uniform(-np.pi/6,np.pi/6), y_direction=np.random.uniform(-np.pi/6,np.pi/6), intercept=np.random.uniform(np.min(h),np.max(h)))
        h = np.where(p!=0,p,h)
    if np.min(h) < 0:
        h = h - np.min(h)
    return h

def terrain2unwrap(surface=np.array(0), wavelength=0.05546576, baseline=123.613815, r=832143.9479681, theta=33.87253381 * math.pi / 180):
    phi = 4 * np.pi / wavelength * baseline / (r * np.sin(theta)) * surface
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

def get_peaks_range(area_width, area_height):
    area_m2 = area_width * area_height
    area_km2 = area_m2 / 1e6
    N = max(3, int(4* np.sqrt(area_km2)))
    peaks_range = (max(2, int(0.6 * N)), max(3, int(1.4 * N)))
    return peaks_range

def generate_set(area_width=2304, area_height=7183.36, set_size=1, range_res=4.5, azimuth_res=14.03):
    results = []
    image_size = ( int(area_height / azimuth_res), int(area_width / range_res))
    pixel_size_x = range_res
    pixel_size_y = azimuth_res
    peaks = get_peaks_range(area_width, area_height)
    variance_scale = 1000
    amplitude_scale = 150
    h = np.zeros([set_size, 1, *image_size])
    PHI = np.zeros([set_size, 1, *image_size])
    phi = np.zeros([set_size, 1, *image_size])
    k = np.zeros([set_size, 1, *image_size])

    for i in range(set_size):
        h[i,0,:,:] = terrain(size=image_size, pixel_size=(pixel_size_x,pixel_size_y), peaks=peaks, variance_scale=variance_scale, amplitude_scale=amplitude_scale, plane=False)
        PHI[i,0,:,:] = terrain2unwrap(surface=h[i,0,:,:], wavelength=0.055, baseline=100, r=800000, theta=30*math.pi/180)
        phi[i,0,:,:] = wrap(unwrapped=PHI[i,0,:,:])
        k[i,0,:,:] = get_k(unwrapped=PHI[i,0,:,:])

    results.append((torch.tensor(h, device=device), torch.tensor(PHI, device=device), torch.tensor(phi, device=device), torch.tensor(k, device=device)))
    return results



results = generate_set(area_width=2304, area_height=7183.36, set_size=1, range_res=4.5, azimuth_res=14.03)
h = results[0][0]
PHI = results[0][1]
phi = results[0][2]

plt.imshow(h[0,0,:,:].cpu())
plt.show()
plt.imshow(PHI[0,0,:,:].cpu(),cmap="gist_rainbow")
plt.show()
plt.imshow(phi[0,0,:,:].cpu(),cmap="gist_rainbow")

x_1, y_1 = np.meshgrid(np.arange(h[0,0,:,:].cpu().shape[1]),np.arange(h[0,0,:,:].cpu().shape[0]))
fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
ax1.plot_surface(x_1,y_1,h[0,0,:,:].cpu(), cmap="terrain",alpha=0.7)
plt.show()

plt.close('all')