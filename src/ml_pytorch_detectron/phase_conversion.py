import numpy as np
import os
import torch
import rasterio
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import random
import geopandas as gpd
from numpy.ma.core import angle
from rasterio import features
import pickle
from skimage.morphology import area_closing
from scipy.special import spence
from shapely.geometry import shape
from pyproj import Transformer, CRS
import fiona

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

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


def get_k(unwrapped):
    k = torch.round(unwrapped/(2*torch.pi)).int()
    return k


def load_gpkg_layer(gpkg_path, layer, crs, bbox):
    xmin, ymin, xmax, ymax = bbox

    try:
        with fiona.open(gpkg_path, layer=layer) as src:
            layer_crs = src.crs_wkt
            layer_crs_obj = CRS.from_wkt(layer_crs)
            raster_crs_obj = CRS.from_user_input(crs)

            if raster_crs_obj != layer_crs_obj:
                transformer = Transformer.from_crs(raster_crs_obj, layer_crs_obj, always_xy=True)
                xs, ys = transformer.transform([xmin, xmax],[ymin, ymax])
                bbox_layer = (min(xs),min(ys),max(xs),max(ys))
            else:
                bbox_layer = bbox

            records = []
            for feat in src.filter(bbox=bbox_layer):
                if feat.get("geometry") is None:
                    continue

                rec = {}
                for k, v in feat.items():
                    if k != "geometry" and k != "type":
                        rec[k] = v

                rec["geometry"] = shape(feat["geometry"])
                records.append(rec)

    except Exception as empty_layer:
        print(f"{empty_layer} load failed.")
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    if not records:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    gdf = gpd.GeoDataFrame(records, crs=layer_crs_obj)

    if gdf.crs != raster_crs_obj:
        gdf = gdf.to_crs(raster_crs_obj)

    return gdf[gdf.geometry.notnull()]


def phase_noise(tensor, metadata, dem_tensor, zabaged_dir, t_days=17.9992, B_perp=123.613815, inc_angle_deg=33.87253381, heading_deg=349.740906, look_side="right", lambda_radar=0.05546576, R=832143.9479681, B_omega=56e6, P_t=4368, A_ant=12.3*0.821, L_a=12.3, tau_p=6.199592966536363e-05, T_sys=800, B_R=56.5e6, pixel_size_x=13.9, pixel_size_y=13.9,):
    device = tensor.device
    tensor_out = tensor.clone()
    sigma_phi2 = torch.zeros_like(tensor, device=device)
    theta_inc_t = torch.tensor(np.deg2rad(inc_angle_deg), device=device)
    heading_rad = np.deg2rad(heading_deg)

    # determine los azimuth
    if look_side.lower() == "right":
        los_azimuth_rad = heading_rad+np.pi/2
    else:
        los_azimuth_rad = heading_rad-np.pi/2

    los_azimuth_rad = torch.tensor(90-los_azimuth_rad, device=device)

    kB = 1.38e-23
    c = 299792458
    # ZABAGED layers
    ZABAGED_LAYERS = {
        "urban": ["BudovaJednotlivaNeboBlokBudov","OstatniPlochaVSidlech","Skladka","PovrchovaTezbaLom","UlozneMisto.shp","KulnaSklenikFoliovnikPristresek","NadzemniZasobniNadrz","RozvalinaZricenina","Hrbitov","ArealUceloveZastavby","Hrad","Zamek","VezovitaStavba","Tribuna","StavebniObjektGIA",],
        "water": ["VodniPlocha","BrehovaCara","BazinaMocal","PozemniNadrz",],
        "shrub": ["LesniPudaSKrovinatymPorostem","Vinice","OvocnySadZahrada","Chmelnice",],
        "grass": ["TrvalyTravniPorost"],
        "crop":  ["OrnaPudaAOstatniDaleNespecifikovanePlochy"],
        "peat":  ["Raseliniste"],}

    ZABAGED_FOREST = "LesniPudaSeStromyKategorizovana"
    ZABAGED_FOREST_2  = "LesniPudaSKosodrevinou"

    HEIGHT_MAP = {"0":10.0, "1":1.25, "2":5.25, "3":14.0, "4":30.0}
    TYPE_CODE  = {"J":1, "L":2, "S":3, "N":4}

    TEMPORAL = {"urban":(0.937, 0.026, 5.056),"grass":(0.666, 0.052, 198),"crop":(1.000, 0.040, 38),"sparse":(0.432, 0.053, 22),"dense":(0.063, 0.025, 1.688),}

    SIGMA0 = {("J", "dense"): {"mean_db": -10.91,"std_db": 1.44},
        ("L","dense"): {"mean_db": -8.86,"std_db": 1.62},
        ("S","dense"): {"mean_db": -9.84,"std_db": 1.18},
        ("N","dense"): {"mean_db": -10.22,"std_db": 2.37},
        ("J","sparse"): {"mean_db": -11.65,"std_db": 1.83},
        ("L","sparse"): {"mean_db": -9.87,"std_db": 1.99},
        ("S","sparse"): {"mean_db": -10.32,"std_db": 1.64},
        ("N","sparse"): {"mean_db": -10.85,"std_db": 2.23},
        ("urban",None): {"mean_db": -7.94,"std_db": 3.28},
        ("water",None): {"mean_db": -18.85,"std_db": 2.53},
        ("shrub",None): {"mean_db": -12.32,"std_db": 2.60},
        ("peat",None): {"mean_db": -11.89,"std_db": 2.96},
        ("crop",None): {"mean_db": -11.87,"std_db": 2.21},
        ("grass",None): {"mean_db": -15.75,"std_db": 5.17},}

    # Cycle through images
    count = 1
    for i in range(tensor.shape[0]):
        meta = metadata[i]
        transform = meta["transform"]
        crs = meta["crs"]
        H, W = tensor.shape[2:]
        x_min = transform[2]
        y_max = transform[5]
        x_max = x_min+transform[0]*W
        y_min = y_max+transform[4]*H
        bbox = (x_min,y_min,x_max,y_max)

        # Create landcover masks
        masks = {}
        for land_cover, files in ZABAGED_LAYERS.items():
            geoms = []
            for fname in files:
                gdf = load_gpkg_layer(zabaged_dir, fname, crs, bbox)
                geoms.extend(gdf.geometry)

            if geoms:
                mask_np = features.rasterize([(g, 1) for g in geoms],out_shape=(H, W),transform=transform,fill=0,dtype="uint8")
                masks[land_cover] = torch.from_numpy(mask_np).to(device)
            else:
                masks[land_cover] = torch.zeros((H,W),dtype=torch.uint8,device=device)

            forest_height = torch.zeros((H,W),device=device)
            forest_type = torch.zeros((H,W),device=device)
            gdf_forest = load_gpkg_layer(zabaged_dir,ZABAGED_FOREST,crs,bbox)
            if not gdf_forest.empty:
                shapes_height = []
                shapes_type = []
                for _, row in gdf_forest.iterrows():
                    properties = row["properties"]
                    key_h = str(properties.get("vyska_k",None))
                    key_t = str(properties.get("druh_k",None))
                    if key_h in HEIGHT_MAP:
                        shapes_height.append((row.geometry,HEIGHT_MAP[key_h]))
                    if key_t in TYPE_CODE:
                        shapes_type.append((row.geometry,TYPE_CODE[key_t]))

                if shapes_height:
                    forest_height = torch.from_numpy(features.rasterize(shapes_height,(H, W),transform=transform,fill=0,dtype="float32")).to(device)

                if shapes_type:
                    forest_type = torch.from_numpy(features.rasterize(shapes_type, (H, W),transform=transform,fill=0,dtype="uint8")).to(device)

            gdf_forest_2 = load_gpkg_layer(zabaged_dir,ZABAGED_FOREST_2,crs,bbox)
            if not gdf_forest_2.empty:
                forest_2 = torch.from_numpy(features.rasterize([(g, 1) for g in gdf_forest_2.geometry],(H, W),transform=transform,fill=0,dtype="uint8")).to(device) == 1
                forest_height[forest_2] = 1.25
                forest_type[forest_2] = TYPE_CODE["J"]

        # Temporal decorrelation
        gamma_temp = torch.ones((H,W),device=device)
        t_days_tensor = torch.tensor(t_days,device=device)
        for name, mask in [("urban", masks["urban"]),("grass", masks["grass"] | masks["peat"]),("crop", masks["crop"])]:
            g0, ginf, tau = TEMPORAL[name]
            gamma_temp[mask==1] = (g0-ginf)*torch.exp(-t_days_tensor/tau)+ginf

        sparse = (forest_height>0)&(forest_height<=5.25)
        dense = forest_height>5.25
        for key, mask in [("sparse", sparse),("dense", dense)]:
            g0, ginf, tau = TEMPORAL[key]
            gamma_temp[mask] = (g0-ginf)*torch.exp(-t_days_tensor/tau)+ginf

        gamma_temp[masks["water"]==1]=0

        # Thermal decorrelation
        sigma0_lc = torch.zeros((H, W),device=device,dtype=torch.float32)

        for typ in ["J","L","S","N"]:
            mask_dense = dense&(forest_type==TYPE_CODE[typ])
            if mask_dense.any():
                cross_section = SIGMA0[(typ,"dense")]
                sigma0_db = cross_section["mean_db"]+torch.randn(mask_dense.sum(),device=device)*cross_section["std_db"]
                sigma0_lc[mask_dense] = 10**(sigma0_db/10)

            mask_sparse = sparse & (forest_type == TYPE_CODE[typ])
            if mask_sparse.any():
                cross_section = SIGMA0[(typ, "sparse")]
                sigma0_db = cross_section["mean_db"] + torch.randn(mask_sparse.sum(), device=device) * cross_section["std_db"]
                sigma0_lc[mask_sparse] = 10** (sigma0_db / 10)

        for land_cover in ["urban","shrub","peat","crop","grass"]:
            mask = masks[land_cover]==1
            if mask.any():
                cross_section = SIGMA0[(land_cover, None)]
                sigma0_db = cross_section["mean_db"] + torch.randn(mask.sum(), device=device) * cross_section["std_db"]
                sigma0_lc[mask] = 10 ** (sigma0_db / 10)

        mask_water = masks["water"]==1
        if mask_water.any():
            cross_section = SIGMA0[("water", None)]
            sigma0_db = cross_section["mean_db"] + torch.randn(mask_water.sum(), device=device) * cross_section["std_db"]
            sigma0_lc[mask_water] = 10 ** (sigma0_db / 10)

        sigma0_lc[sigma0_lc==0] = 1.0
        sigma0_lc = torch.clamp(sigma0_lc,0.000001,None)

        A_scat = lambda_radar*R/L_a*c*tau_p/(2*torch.sin(theta_inc_t))
        G = 4*torch.pi*A_ant/lambda_radar**2
        P_r = (P_t/(4*torch.pi*R**2)*G*A_scat*sigma0_lc*A_ant/(4*torch.pi*R**2))
        P_n = kB*T_sys*B_R
        SNR = P_r/P_n

        gamma_thermal = 1.0/(1.0+1.0/SNR)
        gamma_thermal = torch.clamp(gamma_thermal,0.000001,0.99999)
        gamma_thermal[mask_water] = 0.001

        # Spatial decorrelation
        grad_y, grad_x = torch.gradient(dem_tensor[i, 0],spacing=(pixel_size_y, pixel_size_x))
        slope_los = torch.atan(grad_x*torch.cos(los_azimuth_rad)-grad_y*torch.sin(los_azimuth_rad))
        slope_los = -slope_los
        A = c/(lambda_radar*R*B_omega)
        gamma_spatial = 1-A*B_perp*torch.abs(torch.cos(theta_inc_t-slope_los))

        # Volume decorrelation
        alpha_veg = 2
        h_tensor = forest_height.clone()
        Kz_tensor = 4*torch.pi*B_perp/(R*lambda_radar*torch.sin(theta_inc_t))
        mask_volume = sparse|dense
        alpha_complex = torch.tensor(alpha_veg,dtype=torch.complex64,device=device)
        Kz_complex = torch.tensor(Kz_tensor,dtype=torch.complex64,device=device)
        h_masked = h_tensor[mask_volume]
        prefactor = alpha_complex/(alpha_complex-1j*Kz_complex)
        numerator = torch.exp(-1j*Kz_complex*h_masked)-torch.exp(-alpha_complex*h_masked)
        denominator = 1-torch.exp(-alpha_complex*h_masked)

        gamma_volume_complex = torch.ones((H, W),dtype=torch.complex64,device=device)
        gamma_volume_complex[mask_volume] =prefactor*(numerator/denominator)
        gamma_volume = torch.abs(gamma_volume_complex)


        # Total decorrelation
        gamma_total = torch.clamp(gamma_temp*gamma_thermal*gamma_spatial*gamma_volume,0,0.999)

        def compute_sigma_phi2(gamma):
            gamma_cpu = gamma.detach().cpu().numpy()
            Li2 = spence(1.0-gamma_cpu**2)
            Li2 = torch.from_numpy(Li2).to(device)
            return (torch.pi**2/3-torch.pi*torch.arcsin(gamma)+torch.arcsin(gamma)**2-0.5*Li2)

        sigma_phi2[i, 0] = compute_sigma_phi2(gamma_total)
        noise = torch.sqrt(sigma_phi2[i,0])*torch.randn_like(sigma_phi2[i,0])
        tensor_out[i, 0] = tensor[i,0]+noise
        print(f"Image {count} processed")
        count = count+1
        torch.cuda.empty_cache()

    return tensor_out, sigma_phi2


# h, metadata = tiffs2tensor("D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\dmp_3")
# print(h.shape)
# torch.save(h, "D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\dmp_3\dmp")
# with open('D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\dmp_3\metadata', 'wb') as f:
#     pickle.dump(metadata, f)
# with open('D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\dmp_3\metadata', 'rb') as f:
#     metadata = pickle.load(f)
# h = torch.load("D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\dmp_3\dmp",map_location=device)
# PHI = terrain2unwrap(h[:,:,:,:],wavelength=0.05546576,baseline=129.18913269,r=846227.9829539005+26725/2*2.329562,theta=38.87931758)
# print(PHI.shape)
# torch.save(PHI, r"D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\unwrapped_3\unwrapped")
# k = get_k(PHI)
# torch.save(k, "D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\k\k")
# phi = wrap(PHI)
# torch.save(phi, r"D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\wrapped_3\wrapped")
# dir = r"D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\ZABAGED-3045-gpkg-20251222\ZABAGED_RESULTS.gpkg"
# PHI_noised, variances = phase_noise(PHI, metadata, h, dir,17.9992, 123.613815, 33.87253381, 349.740906,"right",0.05546576, 832143.9479681)
# torch.save(PHI_noised, r"D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\unwrapped_noised\unwrapped_noised")
# torch.save(variances, r"D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\variances\variances")
# phi_noised = wrap(PHI_noised)
# print(phi_noised.shape)
# torch.save(phi_noised, r"D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\wrapped_noised_2\wrapped_noised")
#
#
#
# tensor2tiffs(r"D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\unwrapped_3",PHI,metadata)
# tensor2tiffs(r"D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\k",k,metadata)
# tensor2tiffs(r"D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\wrapped_3",phi,metadata)
# tensor2tiffs(r"D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\wrapped_noised_2",phi_noised,metadata)
# tensor2tiffs(r"D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\variances",variances,metadata)


def remove_boundary_images(tensor,metadata,gdb_path):
    device = tensor.device
    gdf_cz = gpd.read_file(gdb_path,layer="Stat")
    keep = []
    for i in range(tensor.shape[0]):
        meta = metadata[i]
        transform = meta["transform"]
        crs = meta["crs"]
        H, W = tensor.shape[2:]
        if gdf_cz.crs != crs:
            gdf_cz = gdf_cz.to_crs(crs)

        cz_mask = features.rasterize([(geom, 1) for geom in gdf_cz.geometry],out_shape=(H, W),transform=transform,fill=0,dtype="uint8")
        if torch.all(torch.from_numpy(cz_mask).to(device)==1):
            keep.append(i)

    keep = torch.tensor(keep, device=device)
    return tensor[keep], [metadata[i] for i in keep.tolist()]




with open('D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\dmp_3\metadata', 'rb') as f:
    metadata = pickle.load(f)
phi_noised = torch.load("D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\wrapped_noised_2\wrapped_noised",map_location=device)
k = torch.load("D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\k\k",map_location=device)
phi_noised_clipped, metadata_clipped = remove_boundary_images(phi_noised,metadata,gdb_path=r"D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\arccr_4_3\arccr_4_3.gdb")
k_clipped, _ = remove_boundary_images(k,metadata,gdb_path=r"D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\arccr_4_3\arccr_4_3.gdb")
torch.save(phi_noised_clipped,r"D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\wrapped_noised_clipped\wrapped_noised_clipped")
torch.save(k_clipped,r"D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\k_clipped\k_clipped")
with open('D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\dmp_3\metadata_clipped', 'wb') as f:
    pickle.dump(metadata_clipped, f)
tensor2tiffs(r"D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\wrapped_noised_clipped",phi_noised_clipped,metadata_clipped)
tensor2tiffs(r"D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\k_clipped",k_clipped,metadata_clipped)