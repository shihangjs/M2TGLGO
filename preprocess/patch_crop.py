from utils import create_adata
import os
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def extract_patch(image, center, patch_size):
    half_size = int(patch_size // 2)
    x_center, y_center = center
    x_start = max(0, x_center - half_size)
    x_end = min(image.shape[1], x_center + half_size)
    y_start = max(0, y_center - half_size)
    y_end = min(image.shape[0], y_center + half_size)

    patch = image[y_start:y_end, x_start:x_end]
    return Image.fromarray(patch)


def process_spatial_transcriptomics(data_path, dataset_label, image_key="hires", dataset_key="None"):

    output_path = f"{data_path}/{dataset_label}/patches"
    os.makedirs(output_path, exist_ok=True)
    file_path = f"{data_path}/{dataset_label}"

    if dataset_key == "10x":
        adata = sc.read_visium(path=file_path, count_file='filtered_feature_bc_matrix.h5')
        image_path = f"{file_path}/spatial/full_image.tif"
        image = Image.open(image_path)
        image_array = np.array(image)
    else:
        adata, image_array = create_adata(data_path, dataset_label)

    adata.var_names_make_unique()

    spatial_mapping = adata.uns['spatial']
    library_id = list(spatial_mapping.keys())[0]
    spatial_data = spatial_mapping[library_id]
    img_key = "hires"
    spot_size = spatial_data['scalefactors']['spot_diameter_fullres']
    scale_factor = spatial_data['scalefactors'][f"tissue_{img_key}_scalef"]

    coordinates = pd.DataFrame(adata.obsm['spatial'], columns=['fullres_row', 'fullres_col'])
    coordinates['hires_row'] = coordinates['fullres_row'] * scale_factor
    coordinates['hires_col'] = coordinates['fullres_col'] * scale_factor

    patch_size = spot_size
    spot_names = adata.obs.index

    for spot_name, coord in zip(spot_names, coordinates[['fullres_row', 'fullres_col']].values):
        patch_image = extract_patch(
            image_array, (int(coord[0]), int(coord[1])), patch_size)
        patch_save_path = f"{output_path}/{spot_name}.png"
        patch_image.save(patch_save_path)
