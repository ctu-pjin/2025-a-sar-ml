import numpy as np
import os
import torch
import rasterio
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import random
import geopandas as gpd
from rasterio import features
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tiffs2tensor(folder_path):
    tensor_list = []
    metadata_list = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.tif', '.tiff')):
            file_path = os.path.join(folder_path, filename)

            with rasterio.open(file_path) as src:
                data = src.read(1)
                tensor_list.append(torch.from_numpy(data).float())
                metadata_list.append(src.meta.copy())

    tensor_stack = torch.stack(tensor_list)
    tensor_stack = tensor_stack.unsqueeze(1)
    return tensor_stack, metadata_list


def tensor2tiffs(output_folder, tensor_stack, metadata_list):
    os.makedirs(output_folder, exist_ok=True)

    for i, (tensor, meta) in enumerate(zip(tensor_stack, metadata_list)):

        meta = meta.copy()
        meta.update({"count": 1,"dtype": "float32",})

        out_path = os.path.join(output_folder, f"processed_{i+1}.tif")

        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(tensor.squeeze(0).cpu().numpy(), 1)


def terrain2unwrap(surface, wavelength=0.05546576, baseline=123.613815, r=832143.9479681, theta=33.87253381):
    phi = 4 * np.pi / wavelength * baseline / (r *np.sin(np.deg2rad(theta)))*surface
    return phi


def wrap(unwrapped):
    unwrapped_np = unwrapped.cpu().numpy()
    phi = (unwrapped_np + np.pi) % (2*np.pi) - np.pi
    return torch.from_numpy(phi).to(unwrapped.device)


def load_layer(path, target_crs):
    gdf = gpd.read_file(path)
    geom_cols = [col for col in gdf.columns if gdf[col].dtype.name == "geometry"]
    gdf = gdf.set_geometry(geom_cols[0])
    gdf = gdf[gdf.geometry.notnull()]
    gdf = gdf.to_crs(target_crs)
    return gdf


def landcover_noise(tensor,metadata,dir,sigma_buildings=0.3,sigma_vegetation=0.2,sigma_water=0.1):
    SHP = {"build1": f"{dir}/BlokBudov.shp",
           "build2": f"{dir}/ChatovaKolonie.shp",
        "veg1": f"{dir}/Les.shp",
        "veg2": f"{dir}/ZahradaSadParkViniceChmelnice.shp",
        "water": f"{dir}/VodniPlocha.shp"}
    tensor_noised = tensor.clone()
    gdfs = {}

    for key, path in SHP.items():
        gdfs[key] = load_layer(path, metadata[0]["crs"])

    for i in range(tensor.shape[0]):
        transform = metadata[i]["transform"]
        xmin = transform[2]
        ymax = transform[5]
        xmax = xmin+transform[0]*tensor.shape[3]
        ymin = ymax+transform[4]*tensor.shape[2]
        out = tensor[i, 0].cpu().numpy().copy()

        building_mask = np.zeros((tensor.shape[2],tensor.shape[3]),dtype=np.uint8)
        for k in ["build1", "build2"]:
            gdf = gdfs.get(k)
            gdf_clip = gdf.cx[xmin:xmax, ymin:ymax]
            if gdf_clip.empty:
                continue

            building_mask = np.maximum(building_mask,features.rasterize([(geom, 1) for geom in gdf_clip.geometry],out_shape=(tensor.shape[2], tensor.shape[3]),transform=transform,fill=0,dtype=np.uint8))

        if np.any(building_mask):
            noise = np.random.normal(scale=sigma_buildings, size=(tensor.shape[2], tensor.shape[3]))
            out[building_mask==1] += noise[building_mask==1]


        veg_mask = np.zeros((tensor.shape[2], tensor.shape[3]),dtype=np.uint8)
        for k in ["veg1","veg2"]:
            gdf = gdfs.get(k)
            gdf_clip = gdf.cx[xmin:xmax, ymin:ymax]
            if gdf_clip.empty:
                continue

            veg_mask = np.maximum(veg_mask,features.rasterize([(geom, 1) for geom in gdf_clip.geometry],out_shape=(tensor.shape[2], tensor.shape[3]),transform=transform,fill=0,dtype=np.uint8))

        if np.any(veg_mask):
            noise = np.random.normal(scale=sigma_vegetation, size=(tensor.shape[2], tensor.shape[3]))
            out[veg_mask==1] += noise[veg_mask==1]


        water_mask = np.zeros((tensor.shape[2], tensor.shape[3]), dtype=np.uint8)
        gdf_water = gdfs.get("water")
        gdf_clip = gdf_water.cx[xmin:xmax, ymin:ymax]
        if not gdf_clip.empty:
            water_mask = np.maximum(water_mask,features.rasterize([(geom, 1) for geom in gdf_clip.geometry],out_shape=(tensor.shape[2], tensor.shape[3]),transform=transform,fill=0,dtype=np.uint8))

        if np.any(water_mask):
            noise = np.random.normal(scale=sigma_water, size=(tensor.shape[2], tensor.shape[3]))
            out[water_mask==1] += noise[water_mask==1]

        tensor_noised[i, 0] = torch.from_numpy(out).to(tensor.device)

    return tensor_noised








# h, metadata = tiffs2tensor("D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\dmp_3")
# print(h.shape)
# torch.save(h, "D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\dmp_3\dmp")
# with open('D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\dmp_3\metadata', 'wb') as f:
#     pickle.dump(metadata, f)
with open('D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\dmp_3\metadata', 'rb') as f:
    metadata = pickle.load(f)
h = torch.load("D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\dmp_3\dmp",map_location=device)
PHI = terrain2unwrap(h[0:100,:,:,:],wavelength=0.05546576,baseline=129.18913269,r=846227.9829539005+26725/2*2.329562,theta=38.87931758)
print(PHI.shape)
phi = wrap(PHI)
print(phi.shape)
dir = r"D:/Dokumenty/Dokumenty/Skola/CVUT/ml-unwrapping/data50/shp"
phi_noised = landcover_noise(phi, metadata, dir, sigma_buildings=1,sigma_vegetation=1,sigma_water=1)
print(phi_noised.shape)




tensor2tiffs("D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\wrapped_noised",phi_noised,metadata)
tensor2tiffs(r"D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\unwrapped_noised",PHI,metadata)



