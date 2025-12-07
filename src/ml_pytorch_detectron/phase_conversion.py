import numpy as np
import os
import torch
import rasterio
import matplotlib.pyplot as plt

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

def wrap(unwrapped, noise_std=1):
    phi = (unwrapped + np.pi) % (2*np.pi) - np.pi
    if noise_std > 0:
        phi += np.random.normal(scale=np.random.uniform(0.1, 2), size=(unwrapped.shape[2], unwrapped.shape[3]))
    return phi




h, metadata = tiffs2tensor("D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\dmp_3")
print(h.shape)
PHI = terrain2unwrap(h,wavelength=0.05546576,baseline=129.18913269,r=846227.9829539005+26725/2*2.329562,theta=38.87931758)
print(PHI.shape[1:-1])
phi = wrap(PHI,noise_std=0)
print(phi.shape)



plt.figure(figsize=(15, 12))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    if i <5:
        plt.imshow(PHI[i, 0].cpu().numpy(), cmap='viridis')
    else:
        plt.imshow(phi[i-5, 0].cpu().numpy(), cmap='viridis')
    plt.axis('off')

plt.tight_layout()
plt.show()

tensor2tiffs("D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\wrapped_3",phi,metadata)
tensor2tiffs(r"D:\Dokumenty\Dokumenty\Skola\CVUT\ml-unwrapping\dmp1g\unwrapped_3",PHI,metadata)

