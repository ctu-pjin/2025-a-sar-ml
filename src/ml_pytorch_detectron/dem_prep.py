import numpy as np
import requests
import rasterio
import os

def dem_grid(origin,grid_width,grid_height,pix_num,pixel_size,path):
    sampling = int(pix_num * pixel_size)

    x, y = np.meshgrid( np.arange(0, grid_width) * sampling, np.arange(0, grid_height) * sampling )

    X = x + origin[0]
    Y = y + origin[1]

    grid = np.column_stack([X.ravel(), Y.ravel()])


    url = "https://ags.cuzk.cz/arcgis2/rest/services/dmp1g/ImageServer/exportImage"

    name = 0
    for i in range(grid_width * (grid_height - 1)):
        if (i + 1) % grid_width == 0:
            continue

        name += 1

        x_min = grid[i, 0]
        y_min = grid[i, 1]
        x_max = grid[i + grid_width + 1, 0]
        y_max = grid[i + grid_width + 1, 1]

        bbox = f"{x_min},{y_min},{x_max},{y_max}"

        params = {
            "f": "image",
            "bbox": bbox,
            "bboxSR": "32633",
            "size": f"{pix_num},{pix_num}",
            "imageSR": "32633",
            "format": "tiff",
            "pixelType": "F32",
            "noDataInterpretation": "any",
            "noData": "-9999"
        }

        resp = requests.get(url, params=params)

        out_path = path+f"\dem{name}.tiff"
        with open(out_path, "wb") as f:
            f.write(resp.content)



def mask_empty(folder_path, delete_empty=True):

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.tif', '.tiff')):
            continue

        file_path = os.path.join(folder_path, filename)

        with rasterio.open(file_path) as src:
            arr = src.read(1)
            nodata = src.nodata if src.nodata is not None else 0


        mask_empty = arr == nodata
        if delete_empty and np.all(mask_empty):
            os.remove(file_path)
            continue

        arr_masked = np.where(arr == nodata, nodata, arr)

        meta = src.meta.copy()
        meta.update({"dtype": "float32","nodata": nodata,"count": 1})

        with rasterio.open(file_path, "w", **meta) as dst:
            dst.write(arr_masked.astype(np.float32), 1)








# dem_grid((299368.000,5377500.000),72,46,512,13.9,"D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\dmp_3")
mask_empty("D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\dmp_3")
