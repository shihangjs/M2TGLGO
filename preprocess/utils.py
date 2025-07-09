from PIL import Image
from scipy.spatial import cKDTree
import anndata
import scanpy as sc
from scipy.spatial import distance_matrix
import os
import pandas as pd
import numpy as np


def load_and_filter_features(feature_path, sample_names, index_col_name):
    features = pd.read_csv(feature_path)
    features.set_index(index_col_name, inplace=True)
    features = features.iloc[:, :-1]  
    filtered_features = features[features.index.isin(sample_names)]
    filtered_features = filtered_features.loc[sample_names]
    return filtered_features


def calculate_adjacency_matrix(adata, neighbors_num, sample_names):
    spatial_mapping = adata.uns['spatial']
    library_id = list(spatial_mapping.keys())[0]
    spatial_data = spatial_mapping[library_id]

    img_key = "hires"
    spot_size = spatial_data['scalefactors']['spot_diameter_fullres']
    scale_factor = spatial_data['scalefactors'][f"tissue_{img_key}_scalef"]

    coordinates = pd.DataFrame(adata.obsm['spatial'], columns=['fullres_row', 'fullres_col'])
    coordinates.index = adata.obs.index

    filtered_coordinates = coordinates[coordinates.index.isin(sample_names)]
    filtered_coordinates = filtered_coordinates.loc[sample_names]
    dist_matrix = distance_matrix(filtered_coordinates.values, filtered_coordinates.values)

    neighbors_distances = []
    for i in range(dist_matrix.shape[0]):
        sorted_distances = np.sort(dist_matrix[i])
        neighbors_mean_distance = np.mean(sorted_distances[neighbors_num])
        neighbors_distances.append(neighbors_mean_distance)

    threshold = np.mean(neighbors_distances)
    adj_matrix = np.where(dist_matrix < threshold, 1, 0)
    return adj_matrix


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_adata(data_path, dataset_label):
    expression_matrix = pd.read_csv(f"{data_path}/{dataset_label}/{dataset_label}.tsv.gz", sep="\t", index_col=0)
    spot_coordinates = pd.read_csv(f"{data_path}/{dataset_label}/{dataset_label}_selection.tsv.gz", sep="\t")
    spot_coordinates['spot'] = spot_coordinates['x'].astype( str) + 'x' + spot_coordinates['y'].astype(str)
    common_spots = expression_matrix.index.intersection(spot_coordinates['spot'])
    expression_matrix = expression_matrix.loc[common_spots]
    spot_coordinates = spot_coordinates.set_index('spot').loc[common_spots].reset_index()
    coordinates = spot_coordinates[['pixel_x', 'pixel_y']].values
    tree = cKDTree(coordinates)
    distances, _ = tree.query(coordinates, k=5)
    min_distances = distances[:, 1:].min(axis=1)
    spot_coordinates['neighbor_distance'] = min_distances
    image_path = f"{data_path}/{dataset_label}/{dataset_label}.jpg"
    image = Image.open(image_path)
    image_array = np.array(image)
    image_data = {
        dataset_label: {
            'images': {
                'hires': image_array,
                'lowres': image_array
            },
            'scalefactors': {
                'spot_diameter_fullres': np.min(min_distances),
                'tissue_hires_scalef': 1,
                'tissue_lowres_scalef': 1
            },
            'metadata': {
                'chemistry_description': "raw data not provided",
                'software_version': 'raw data not provided'
            }
        }
    }
    adata = anndata.AnnData(X=expression_matrix)
    adata.obsm['spatial'] = spot_coordinates[['pixel_x', 'pixel_y']].values
    adata.obs['neighbor_distance'] = spot_coordinates['neighbor_distance'].values
    adata.uns['spatial'] = image_data

    return adata, image_array