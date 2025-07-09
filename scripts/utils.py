import json
from typing import List, Dict, Set
import logging
from typing import Dict, List
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from typing import List, Dict
import os
import re
import pickle
import networkx as nx
import multiprocessing as mp
from typing import List, Dict, Set, Tuple, Union, Optional

import pandas as pd
import numpy as np
import torch
import dgl
import anndata
import scanpy as sc
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from scipy.spatial import distance_matrix
from scipy.spatial import cKDTree
from goatools.obo_parser import GODag
from sklearn.neighbors import NearestNeighbors

from resnet import ResNetFeatureExtractor
from calculate_gene_similarity import *
Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("sh/scripts/MultiDimGCN/log/utils.log")
    ]
)
logger = logging.getLogger(__name__)


def scale_features(filtered_features, axis, method, log_transform):
    if not isinstance(filtered_features, (pd.DataFrame, np.ndarray, torch.Tensor)):
        raise TypeError("data type must be pandas DataFrame or numpy ndarray。")

    if method not in {"normalize", "standardize"}:
        raise ValueError("method must be 'normalize' or 'standardize'。")

    if isinstance(filtered_features, pd.DataFrame):
        if method == "normalize":
            total_sum = filtered_features.sum(axis=axis)
            total_sum[total_sum == 0] = 1
            scaled_features = filtered_features.div(total_sum, axis=1 - axis)
        else:
            mean_values = filtered_features.mean(axis=axis)
            std_devs = filtered_features.std(axis=axis)
            std_devs[std_devs == 0] = 1
            scaled_features = filtered_features.sub(mean_values, axis=1 - axis).div(std_devs, axis=1 - axis)

        if log_transform:
            scaled_features = np.log1p(scaled_features)

        return scaled_features

    elif isinstance(filtered_features, np.ndarray):
        
        if method == "normalize":
            total_sum = filtered_features.sum(axis=axis, keepdims=True)
            total_sum[total_sum == 0] = 1
            scaled_features = filtered_features / total_sum
        else:
            mean_values = filtered_features.mean(axis=axis, keepdims=True)
            std_devs = filtered_features.std(axis=axis, keepdims=True)
            std_devs[std_devs == 0] = 1
            scaled_features = (filtered_features - mean_values) / std_devs
            
        if log_transform:
            scaled_features = np.log1p(scaled_features)

    elif isinstance(filtered_features, torch.Tensor):
        
        if method == "normalize":
            total_sum = filtered_features.sum(dim=axis, keepdim=True)
            total_sum[total_sum == 0] = 1
            scaled_features = filtered_features / total_sum
        else:
            mean_values = filtered_features.mean(dim=axis, keepdim=True)
            std_devs = filtered_features.std(dim=axis, keepdim=True)
            std_devs[std_devs == 0] = 1
            scaled_features = (filtered_features - mean_values) / std_devs

        if log_transform:
            scaled_features = torch.log1p(scaled_features)

        return scaled_features


def load_and_filter_features(feature_path, spot_used, index_col_name, columns_to_use):
    try:
        features = pd.read_csv(feature_path)
        features.set_index(index_col_name, inplace=True)

        if features.isna().any().any():
            logger.info("Warning: The data contains NaN values, which have been replaced with 0.")
            features.fillna(0, inplace=True)
            
        if columns_to_use:
            missing_columns = set(columns_to_use) - set(features.columns)
            if missing_columns:
                raise ValueError(f"The following columns are missing from the feature matrix: {missing_columns}")
            features = features[columns_to_use]

        if set(spot_used).issubset(features.index):
            filtered_features = features.loc[spot_used]
        else:
            new_dataframe = pd.DataFrame(0, index=spot_used, columns=features.columns)
            valid_spots = features.index.intersection(spot_used)
            new_dataframe.loc[valid_spots] = features.loc[valid_spots]
            missing_spots = set(spot_used) - set(features.index)
            logger.info(f"{len(missing_spots)} missing spot features detected in the current file — filled with 0.")
            filtered_features = new_dataframe

        return scale_features(filtered_features, axis=0, method="standardize", log_transform=False)

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Feature file not found: {feature_path}") from e
    except ValueError as e:
        raise ValueError(f"An error occurred while processing the feature matrix: {str(e)}") from e


def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)

    adj_normalized = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

    return adj_normalized


def get_cumulative_adjacency_matrix(adjacency_matrices, max_hop):
    cumulative_adj_matrix = np.zeros_like(
        next(iter(adjacency_matrices.values())))

    for hop in range(1, max_hop + 1):
        cumulative_adj_matrix += adjacency_matrices[hop]
    cumulative_adj_matrix = (cumulative_adj_matrix > 0).astype(int)
    return cumulative_adj_matrix


def calculate_adjacency_matrix(adata, spot_used, neighbors_num, max_hop, device):
    if 'spatial' not in adata.obsm:
        raise ValueError("'spatial' data is not found in adata.obsm.")
    coordinates = pd.DataFrame(adata.obsm['spatial'], index=adata.obs.index)
    try:
        filtered_coordinates = coordinates.loc[spot_used].values
    except KeyError as e:
        raise ValueError(f"Some spot IDs are missing from `adata.obs`: {e}")
    num_nodes = filtered_coordinates.shape[0]
    nbrs = NearestNeighbors(n_neighbors=neighbors_num + 1,
                            algorithm='auto').fit(filtered_coordinates)
    distances, indices = nbrs.kneighbors(filtered_coordinates)

    neighbor_indices = indices[:, 1:].flatten()
    row_indices = np.repeat(np.arange(num_nodes), neighbors_num)
    data = np.ones(len(neighbor_indices), dtype=int)

    adjacency_matrix_1hop = csr_matrix((data, (row_indices, neighbor_indices)), shape=(num_nodes, num_nodes))
    G = nx.from_scipy_sparse_array(adjacency_matrix_1hop)
    if nx.is_directed(G):
        logger.info("The current adjacency matrix represents an undirected graph.")
    distances_matrix = shortest_path(csgraph=adjacency_matrix_1hop, directed=False, unweighted=True)
    distances_matrix[np.isinf(distances_matrix)] = max_hop + 1

    adjacency_matrices: Dict[int, np.ndarray] = {}
    for hop in range(1, max_hop + 1):
        adjacency_matrices[hop] = (distances_matrix == hop).astype(int)

    cumulative_adj_matrix = get_cumulative_adjacency_matrix(adjacency_matrices, max_hop)
    adjacency_matrices["cumulative_adj_matrix"] = cumulative_adj_matrix

    for key, value in list(adjacency_matrices.items()):
        if isinstance(key, int):
            tensor_key = f'adj_matrix_{key}hop_tensor'
            adjacency_matrices[tensor_key] = torch.tensor(value, dtype=torch.float32).to(device)
            del adjacency_matrices[key]
        elif key == "cumulative_adj_matrix":
            adjacency_matrices[key] = torch.tensor(value, dtype=torch.float32).to(device)

    return adjacency_matrices


def convert_adj_matrix_to_dgl_graph(adjacency_matrix, gnn_type, device):
    self_loop = torch.diag(adjacency_matrix).sum().item() > 0
    edge_list = torch.nonzero(adjacency_matrix, as_tuple=False).t()
    graph = dgl.graph((edge_list[0], edge_list[1]), device=device)
    if not self_loop:
        graph = dgl.add_self_loop(graph)
    return graph


def create_adata(data_path, dataset_label):

    expression_matrix = pd.read_csv(f"{data_path}/{dataset_label}/{dataset_label}.tsv.gz", sep="\t", index_col=0)
    spot_coordinates = pd.read_csv(f"{data_path}/{dataset_label}/{dataset_label}_selection.tsv.gz", sep="\t")

    spot_coordinates['spot'] = (spot_coordinates['x'].astype(str) + 'x' + spot_coordinates['y'].astype(str))

    common_spots = expression_matrix.index.intersection(spot_coordinates['spot'])
    expression_matrix = expression_matrix.loc[common_spots]
    spot_coordinates = (spot_coordinates.set_index('spot').loc[common_spots].reset_index())

    if 'selected' not in spot_coordinates.columns:
        spot_coordinates['selected'] = 1
    spot_coordinates_df = spot_coordinates[['index', 'selected', 'new_x', 'new_y', 'pixel_x', 'pixel_y']]
    spot_coordinates_df = spot_coordinates_df.rename(columns={'index': 'barcodes'})

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
    supp_info = spot_coordinates[['x', 'y', 'new_x', 'new_y']]
    adata.obsm['supp_info'] = supp_info.values
    adata.obsm['spatial'] = spot_coordinates[['pixel_x', 'pixel_y']].values
    adata.obs['neighbor_distance'] = spot_coordinates['neighbor_distance'].values
    adata.uns['spatial'] = image_data

    return adata, image_array


def plot_loss_curve(loss_values, label, title, color, file_name):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label=label, color=color)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.savefig(file_name)
    plt.close()


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def check_preprocess_data(input_dir, dataset_key, dataset_labels, train_dataset_labels, gene_type, n_top_genes, gene_embedding_keys, marker_genes, gnn_type, mode, device):
    features_pkl_path = f'{input_dir}/{gene_type}_preprocess_output/1.features_and_adjacencies_per_dataset.pkl'
    spots_pkl_path = f'{input_dir}/{gene_type}_preprocess_output/3.spots_used_per_dataset.pkl'
    analyzed_genes_pkl_path = f'{input_dir}/{gene_type}_preprocess_output/4.shared_{gene_type}_genes.pkl'

    if (os.path.exists(features_pkl_path) and os.path.exists(spots_pkl_path) and os.path.exists(analyzed_genes_pkl_path)):
        logger.info("Preprocessed file found. Loading existing data...")
        
        with open(features_pkl_path, 'rb') as f:
            features_and_adjacencies_per_dataset = pickle.load(f)

        with open(analyzed_genes_pkl_path, 'rb') as f:
            analyzed_genes = pickle.load(f)

        with open(spots_pkl_path, 'rb') as f:
            spots_used_per_dataset = pickle.load(f)

        for dataset_label in features_and_adjacencies_per_dataset:
            for key, value in features_and_adjacencies_per_dataset[dataset_label].items():
                if isinstance(value, torch.Tensor):
                    features_and_adjacencies_per_dataset[dataset_label][key] = value.to(device)
        logger.info("Preprocessed file loaded successfully!")

    else:
        logger.info("No preprocessed file found. Starting data preprocessing...")

        features_and_adjacencies_per_dataset, analyzed_genes, spots_used_per_dataset = preprocess_data(
            input_dir, dataset_key, dataset_labels, train_dataset_labels, 
            gene_type, n_top_genes, gene_embedding_keys, marker_genes, gnn_type, mode, device
        )
        logger.info("Data preprocessing finished and saved successfully.")

    return features_and_adjacencies_per_dataset, analyzed_genes, spots_used_per_dataset


def preprocess_data(input_dir, dataset_key, dataset_labels, train_dataset_labels, gene_type, n_top_genes, gene_embedding_keys, marker_genes, gnn_type, mode, device):
    features_and_adjacencies_per_dataset = {}
    spots_used_per_dataset = {}
    adata_list = []

    for dataset_label in dataset_labels:
        logger.info(dataset_label)
        subfolder_path = f'{input_dir}/{dataset_label}'
        output_dir = f"{subfolder_path}/{gene_type}_preprocess_output"
        create_dir(output_dir)
        create_dir(f"{subfolder_path}/{gene_type}_data")

        min_segment_cell_threshold = 0
        cell_count_path = f'{subfolder_path}/feature/cell_count.csv'
        cell_count = pd.read_csv(cell_count_path)
        filtered_samples = cell_count[cell_count['cell_count'] >= min_segment_cell_threshold]
        spot_used = filtered_samples['file_name'].tolist()
        pd.DataFrame(spot_used).to_csv(f"{output_dir}/spot_used.csv")
        spots_used_per_dataset[dataset_label] = spot_used

        if dataset_key == "10x":
            adata = sc.read_visium(path=subfolder_path, count_file='filtered_feature_bc_matrix.h5')
        else:
            adata, _ = create_adata(input_dir, dataset_label)

        adata.var_names_make_unique()
        duplicated_vars = adata.var_names[adata.var_names.duplicated()]
        if duplicated_vars.any(): 
            raise ValueError(f"Duplicate variable names found in the current adata object: {duplicated_vars}")

        adata_list.append(adata)

        if dataset_label in train_dataset_labels:
 
            cell_level_features_path = (f'{subfolder_path}/feature/cell_level_features.csv')

            cell_level_features_used = [
                "Area", "Perimeter", "Eccentricity", "MajorAxisLength", "MinorAxisLength", "Solidity", "Extent",
                "Orientation", "Circularity", "AspectRatio", "ConvexArea", "Convexity",
                "SkeletonLength", "ConvexPerimeter"
            ]

            filtered_cell_level_features = load_and_filter_features(cell_level_features_path, spot_used, "file_name", cell_level_features_used)
            patch_level_features_used = ["NearestNeighborDistance",
                                         "AverageNeighborDistance",
                                         "FurthestNeighborDistance"
                                         ]
            filtered_patch_level_features = load_and_filter_features(cell_level_features_path, spot_used, "file_name", patch_level_features_used)

            filtered_patch_level_features_tensor = torch.tensor(filtered_patch_level_features.values, dtype=torch.float32).to(device)
            filtered_cell_level_features_tensor = torch.tensor(filtered_cell_level_features.values, dtype=torch.float32).to(device)

            max_hop = 3
            neighbors_num_1_hop = 6
            adjacency_matrices = calculate_adjacency_matrix(adata, spot_used, neighbors_num_1_hop, max_hop, device)

            features_and_adjacencies_per_dataset[dataset_label] = {
                'filtered_patch_level_features_tensor': filtered_patch_level_features_tensor,
                'filtered_cell_level_features_tensor': filtered_cell_level_features_tensor,
                **adjacency_matrices
            }

    if mode == "train":
        adata_integrated = sc.concat(adata_list, label='batch', keys=dataset_labels)
        sc.pp.normalize_total(adata_integrated, target_sum=1e4)
        sc.pp.log1p(adata_integrated)
        sc.pp.highly_variable_genes(adata_integrated, flavor="seurat_v3", n_top_genes=n_top_genes)

        # if marker_genes:
        #     highly_variable_genes.extend(marker_genes)
        # highly_variable_genes = list(set(gene_embedding_keys).intersection(highly_variable_genes))
        # pattern = re.compile(r"^(AC|AL|LINC|MT|RPL|RPS|C[0-9]+orf)")
        # highly_variable_genes = [
        #     gene for gene in highly_variable_genes if not pattern.match(gene)
        # ]
        # shared_highly_variable_genes = set(highly_variable_genes)
        # for adata in adata_list:
        #     current_genes = set(adata.var_names)
        #     shared_highly_variable_genes = shared_highly_variable_genes.intersection(current_genes)
        # missing_genes = [gene for gene in marker_genes if gene not in shared_highly_variable_genes]
 

        if gene_type == "highly_variable":
            highly_variable_genes = list(adata_integrated.var.index[adata_integrated.var["highly_variable"]])
            analyzed_genes, _ = filter_and_merge_shared_genes(adata_list, gene_type, highly_variable_genes, gene_embedding_keys, marker_genes)

        elif gene_type == "highly_expressed":
            if dataset_key == "10x":
                gene_expression = pd.DataFrame(adata_integrated.X.toarray())
            elif dataset_key == "other":
                gene_expression = pd.DataFrame(adata_integrated.X)
            gene_expression.index = adata_integrated.obs_names
            gene_expression.columns = adata_integrated.var_names

            mean_gene_expression = gene_expression.mean(axis=0)
            highly_expressed_genes = mean_gene_expression.sort_values(ascending=False).head(3000)

            genes = highly_expressed_genes.index.to_list()
            analyzed_genes, missing_genes = filter_and_merge_shared_genes(adata_list, gene_type, genes, gene_embedding_keys, marker_genes)
            analyzed_genes = highly_expressed_genes[highly_expressed_genes.index.isin(analyzed_genes)].index.to_list()

        create_dir(f'{input_dir}/{gene_type}_preprocess_output')

        with open(f'{input_dir}/{gene_type}_preprocess_output/1.features_and_adjacencies_per_dataset.pkl', 'wb') as file:
            pickle.dump(features_and_adjacencies_per_dataset, file)

        with open(f'{input_dir}/{gene_type}_preprocess_output/3.spots_used_per_dataset.pkl', 'wb') as file:
            pickle.dump(spots_used_per_dataset, file)

        with open(f'{input_dir}/{gene_type}_preprocess_output/4.shared_{gene_type}_genes.pkl', 'wb') as file:
            pickle.dump(analyzed_genes, file)

        with open(f"{input_dir}/{gene_type}_preprocess_output/4.shared_{gene_type}_genes.txt", 'w') as f:
            for gene in analyzed_genes:
                f.write(f"{gene}\n")

        output_path = f"{input_dir}/{gene_type}_preprocess_output/4.shared_{gene_type}_genes.npy"
        np.save(output_path, np.array(analyzed_genes))

        return features_and_adjacencies_per_dataset, analyzed_genes, spots_used_per_dataset
    else:
        return features_and_adjacencies_per_dataset, spots_used_per_dataset


def filter_and_merge_shared_genes(adata_list, gene_type, initial_genes, gene_embedding_keys, marker_genes):
    if marker_genes:
        logger.info("Merging marker genes with the initial gene list.")
        all_genes = initial_genes + marker_genes
    else:
        all_genes = initial_genes
        
    filtered_genes = list(set(gene_embedding_keys).intersection(all_genes))
    pattern = re.compile(r"^(AC|AL|LINC|MT|RPL|RPS|C[0-9]+orf)")
    filtered_genes = [gene for gene in filtered_genes if not pattern.match(gene)]\
    
    shared_genes = set(filtered_genes)
    for adata in adata_list:
        current_genes = set(adata.var_names)
        shared_genes = shared_genes.intersection(current_genes)
    logger.info(f"The number of {gene_type} genes is: {len(shared_genes)}")

    missing_genes = []
    if marker_genes:
        missing_genes = [
            gene for gene in marker_genes if gene not in shared_genes]
        if missing_genes:
            logger.info(f"The following marker genes are missing during {gene_type}_gene selection: {missing_genes}")

    return list(shared_genes), missing_genes


def check_image_feature(input_dir, dataset_key, dataset_labels, train_dataset_labels, spots_used_per_dataset, gene_type, mode, device):

    image_feature_path = f'{input_dir}/{gene_type}_preprocess_output/2.img_features_per_dataset.pkl'

    if os.path.exists(image_feature_path):
        logger.info(f"File {image_feature_path} found. Loading data...")
        with open(image_feature_path, 'rb') as f:
            img_features_per_dataset = pickle.load(f)

        for dataset_label in img_features_per_dataset:
            img_features_per_dataset[dataset_label] = torch.tensor(img_features_per_dataset[dataset_label], dtype=torch.float32).to(device)
        logger.info("Image features loaded successfully!")
    else:
        logger.info(f"File {image_feature_path} not found. Generating image features...")
        img_features_per_dataset = get_image_feature(input_dir, dataset_key, dataset_labels, train_dataset_labels, spots_used_per_dataset, gene_type, mode, device)
        logger.info("Image feature extraction completed!")
    return img_features_per_dataset


def get_image_feature(input_dir, dataset_key, dataset_labels, train_dataset_labels, spots_used_per_dataset, gene_type, mode, device):
    img_features_per_dataset = {}

    for dataset_label in dataset_labels:
        print(dataset_label)

        subfolder_path = f'{input_dir}/{dataset_label}'
        patches_dir = f"{subfolder_path}/patches"

        spot_used = spots_used_per_dataset[dataset_label]

        if dataset_key == "10x":
            adata = sc.read_visium(path=subfolder_path, count_file='filtered_feature_bc_matrix.h5')
        else:
            adata, _ = create_adata(input_dir, dataset_label)

        adata.var_names_make_unique()
        duplicated_vars = adata.var_names[adata.var_names.duplicated()]
        if duplicated_vars.any():  
            raise ValueError(f"Duplicate variable names found in the current adata object: {duplicated_vars}")

        if dataset_label in train_dataset_labels:

            img_features = []
            model_resnet = ResNetFeatureExtractor(device).to(device)
            for img_name in tqdm(spot_used, desc=f"Extracting image features for {dataset_label}"):
                img_path = f"{patches_dir}/{img_name}.png"
                img = Image.open(img_path)
                feature = model_resnet(img).squeeze()
                img_features.append(feature)
            img_features_tensor = torch.stack(img_features).to(device)
            img_features_scaled = scale_features(img_features_tensor, axis=0, method="standardize", log_transform=False)
            img_features_per_dataset[dataset_label] = img_features_scaled

    if mode == "train":
        create_dir(f'{input_dir}/{gene_type}_preprocess_output')
        with open(f'{input_dir}/{gene_type}_preprocess_output/2.img_features_per_dataset.pkl', 'wb') as file:
            pickle.dump(img_features_per_dataset, file)

    return img_features_per_dataset


def check_gene_expression(input_dir, dataset_key, train_dataset_labels, spots_used_per_dataset, gene_type, analyzed_genes, mode, device,):
    gene_expression_path = f'{input_dir}/{gene_type}_preprocess_output/4.raw_{gene_type}_gene_expression_data.pkl'
    if os.path.exists(gene_expression_path):
        logger.info(f"File {gene_expression_path} found. Loading gene expression data...")
        
        with open(gene_expression_path, 'rb') as f:
            raw_gene_expression_data = pickle.load(f)

        for dataset_label in raw_gene_expression_data:
            raw_gene_expression_data[dataset_label] = torch.tensor(raw_gene_expression_data[dataset_label], dtype=torch.float32).to(device)
        logger.info("Gene expression matrix loaded successfully!")

    else:
        logger.info(f"File {gene_expression_path} not found. Generating gene expression data...")
        raw_gene_expression_data = get_gene_expression(input_dir, dataset_key, train_dataset_labels, spots_used_per_dataset, gene_type, analyzed_genes, mode, device)
        logger.info("Gene expression matrix extracted successfully!")
    return raw_gene_expression_data


def get_gene_expression(input_dir, dataset_key, dataset_labels, spots_used_per_dataset, gene_type, analyzed_genes, mode, device,):
    raw_gene_expression_data = {}
    valid_genes_consistent = True 
    for dataset_label in dataset_labels:
        print(dataset_label)
        subfolder_path = f'{input_dir}/{dataset_label}'
        create_dir(f"{subfolder_path}/data")

        if dataset_key == "10x":
            adata = sc.read_visium(path=subfolder_path, count_file='filtered_feature_bc_matrix.h5')
        else:
            adata, _ = create_adata(input_dir, dataset_label)

        adata.var_names_make_unique()

        duplicated_vars = adata.var_names[adata.var_names.duplicated()]
        if duplicated_vars.any():  
            raise ValueError(f"Duplicate variable names found in the current adata object: {duplicated_vars}")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        genes_not_in_adata = [gene for gene in analyzed_genes if gene not in adata.var_names]

        if genes_not_in_adata:
            logger.info(f"In dataset {dataset_label}, the following genes were not found in the adata object: {genes_not_in_adata}")
            valid_genes_consistent = False 

        if valid_genes_consistent:
            valid_genes = list(analyzed_genes)
        else:
            valid_genes = [gene for gene in analyzed_genes if gene in adata.var_names]

        spot_used = spots_used_per_dataset[dataset_label]
        adata_hvgs = adata[:, valid_genes]  
        adata_hvgs = adata_hvgs[adata_hvgs.obs_names.isin(spot_used), :]  

        indices = [adata_hvgs.obs_names.tolist().index(spot) for spot in spot_used if spot in adata_hvgs.obs_names]
        adata_hvgs = adata_hvgs[indices, :]  

        filtered_gene_expression_data = adata_hvgs.X.toarray()
        raw_gene_expression_data[dataset_label] = torch.tensor(filtered_gene_expression_data, dtype=torch.float32).to(device)

    if mode == "train":
        with open(f'{input_dir}/{gene_type}_preprocess_output/4.raw_{gene_type}_gene_expression_data.pkl', 'wb') as f:
            pickle.dump(raw_gene_expression_data, f)
        return raw_gene_expression_data
    elif mode == "eval":
        return raw_gene_expression_data, adata_hvgs


def check_gene_similarity_matrix(input_dir, species, gene_type, analyzed_genes, GODag_path, raw_gene_simslarity_matrix_path="",):

    gene_similarity_matrix_path = f'{input_dir}/{gene_type}_preprocess_output/5.{gene_type}_gene_similarity_matrix.csv'
    gene_similarity_matrix_pkl_path = f'{input_dir}/{gene_type}_preprocess_output/5.{gene_type}_gene_similarity_results.pkl'

    if os.path.exists(gene_similarity_matrix_path) and os.path.exists(gene_similarity_matrix_pkl_path):
    
        logger.info(f"The gene similarity matrix file don't exist. Loading gene similarity matrix from: {gene_similarity_matrix_path}")
        gene_similarity_matrix = pd.read_csv(gene_similarity_matrix_path, index_col=0)
        logger.info("Gene similarity matrix loaded successfully!")
        
    elif os.path.exists(raw_gene_simslarity_matrix_path):
        logger.info(f"The gene similarity matrix file already exists. Loading gene similarity matrix from: {gene_similarity_matrix_path}")

        raw_gene_similarity_matrix = pd.read_csv(raw_gene_simslarity_matrix_path)
        gene_similarity_matrix = raw_gene_similarity_matrix.reindex(index=analyzed_genes, columns=analyzed_genes)
        gene_similarity_matrix.to_csv( f'{input_dir}/{gene_type}_preprocess_output/5.{gene_type}_gene_similarity_matrix.csv')
        logger.info("Finished extracting gene similarity matrix!")

    else:
        logger.info("The gene_similarity_matrix file was not found. Beginning computation of the gene similarity matrix...")
        gene_similarity_matrix = compute_gene_adjacency_matrix(input_dir, species, gene_type, analyzed_genes, GODag_path)
        logger.info("Gene similarity matrix computation finished successfully!")

    return gene_similarity_matrix


def compute_gene_adjacency_matrix(input_dir, species, gene_type, analyzed_genes, GODag_path, num_workers_fetch=80, num_workers_compute=None):
    output_dir = os.path.join(input_dir, f'{gene_type}_preprocess_output')
    create_dir(output_dir)
    logger.info("Loading GO DAG...")
    go_dag = GODag(GODag_path)

    logger.info("Fetching Ensembl IDs...")
    ensembl_ids_mapping, missing_gene = get_ensembl_ids(list(analyzed_genes), species)

    all_ensembl_ids = set(
        ens_id for gene, ens_ids in ensembl_ids_mapping.items()
        for ens_id in ens_ids if ens_id != 'Not Found'
    )

    logger.info("Retrieving all GO terms in advance...")
    go_terms_mapping = prefetch_go_terms(list(all_ensembl_ids), num_workers=num_workers_fetch)

    valid_genes = [gene for gene in analyzed_genes if ensembl_ids_mapping[gene] and 'Not Found' not in ensembl_ids_mapping[gene]]
    invalid_genes = list(set(analyzed_genes) - set(valid_genes))
    if invalid_genes:
        logger.info(f"Ensembl IDs not found for the following genes; similarity with other genes will be set to 0: {invalid_genes}")

    gene_pairs = list(combinations(valid_genes, 2))
    logger.info(f"A total of {len(gene_pairs)} valid gene pairs were generated.")

    if num_workers_compute is None:
        num_workers_compute = mp.cpu_count() - 2
    logger.info(f"Number of cores used for similarity computation: {num_workers_compute}")

    gene_similarity_matrix = compute_all_gene_similarities_parallel(
        gene_pairs,
        ensembl_ids_mapping,
        go_terms_mapping,
        go_dag,
        output_dir,
        num_workers=num_workers_compute
    )

    for gene in invalid_genes:
        gene_similarity_matrix[gene] = {other_gene: 0.0 for other_gene in analyzed_genes}
        for other_gene in analyzed_genes:
            gene_similarity_matrix[other_gene][gene] = 0.0

    pickle_file_path = os.path.join(output_dir, f'5.{gene_type}_gene_similarity_results.pkl')
    with open(pickle_file_path, 'wb') as pklfile:
        pickle.dump(gene_similarity_matrix, pklfile)
    logger.info(f"Gene similarity results have been saved to {pickle_file_path}")


    gene_similarity_matrix_df = pd.DataFrame(gene_similarity_matrix).fillna(0)
    csv_file_path = os.path.join(
        output_dir, f'5.{gene_type}_gene_similarity_matrix.csv')
    gene_similarity_matrix_df.to_csv(csv_file_path)

    return gene_similarity_matrix_df


def process_gene_similarity_matrix(analyzed_genes, gene_similarity_matrix, device, threshold: float = 0.75):
    missing_genes = [gene for gene in analyzed_genes if gene not in gene_similarity_matrix.index]
    if missing_genes:
        logger.info(f"The following genes are missing from the rows or columns of the gene_similarity_matrix: {missing_genes}")
        raise ValueError(f"Error: Some genes are missing. Missing genes: {missing_genes}. Execution aborted.")
    gene_similarity_matrix_reordered = gene_similarity_matrix.reindex(
        index=analyzed_genes,
        columns=analyzed_genes
    )

    gene_similarity_matrix_np = gene_similarity_matrix_reordered.to_numpy()

    gene_similarity_matrix_np = np.where(gene_similarity_matrix_np > threshold, 1, 0)
    np.fill_diagonal(gene_similarity_matrix_np, 1)
    gene_similarity_matrix_tensor = torch.tensor(gene_similarity_matrix_np, dtype=torch.float32).to(device)

    return gene_similarity_matrix_tensor


def create_comp_data(input_dir, dataset_key, dataset_labels, spots_used_per_dataset, gene_type, analyzed_genes, mode, device,):
    valid_genes_consistent = True 
    for dataset_label in dataset_labels:
        print(dataset_label)
        subfolder_path = f'{input_dir}/{dataset_label}'
        create_dir(f"{subfolder_path}/data")
        if dataset_key == "10x":
            adata = sc.read_visium(path=subfolder_path, count_file='filtered_feature_bc_matrix.h5')
        else:
            adata, _ = create_adata(input_dir, dataset_label)
        adata.var_names_make_unique()
        duplicated_vars = adata.var_names[adata.var_names.duplicated()]
        if duplicated_vars.any(): 
            raise ValueError(f"Duplicate variable names found in the current adata object: {duplicated_vars}")

        genes_not_in_adata = [gene for gene in analyzed_genes if gene not in adata.var_names]

        if genes_not_in_adata:
            logger.info(f"The following genes in dataset {dataset_label} were not found in adata: {genes_not_in_adata}")
            valid_genes_consistent = False  

        if valid_genes_consistent:
            valid_genes = list(analyzed_genes)
        else:
            valid_genes = [gene for gene in analyzed_genes if gene in adata.var_names]
            
        spot_used = spots_used_per_dataset[dataset_label]
        adata_filtered = adata[:, valid_genes]  
        adata_filtered = adata_filtered[adata_filtered.obs_names.isin(spot_used), :]  

        indices = [adata_filtered.obs_names.tolist().index(spot) for spot in spot_used if spot in adata_filtered.obs_names]
        adata_filtered = adata_filtered[indices, :]

        if dataset_key == "10x":
            df = pd.DataFrame({
                'x': adata_filtered.obs['array_row'].values,
                'y': adata_filtered.obs['array_col'].values,
                'new_x': adata_filtered.obs['array_row'].values,
                'new_y': adata_filtered.obs['array_col'].values,
                'pixel_x': adata_filtered.obsm['spatial'][:, 0], 
                'pixel_y': adata_filtered.obsm['spatial'][:, 1]  
            })
        else:
            df = pd.DataFrame({
                'x': adata_filtered.obsm['supp_info'][:, 0],
                'y': adata_filtered.obsm['supp_info'][:, 1],
                'new_x': adata_filtered.obsm['supp_info'][:, 2],
                'new_y': adata_filtered.obsm['supp_info'][:, 3],
                'pixel_x': adata_filtered.obsm['spatial'][:, 0], 
                'pixel_y': adata_filtered.obsm['spatial'][:, 1]  
            })

        df['x'] = df['x'].astype(int)
        df['y'] = df['y'].astype(int)
        index = (df['x'].astype(int).astype(str) + 'x' + df['y'].astype(int).astype(str))
        df['selected'] = 1 

        mapping_df = pd.DataFrame({
            'old_name': adata_filtered.obs_names.tolist(),
            'new_name': index
        })
        mapping_df.to_csv(f"{subfolder_path}/mapping_df.csv", index=False)
        import shutil

        source_patches_dir = os.path.join(subfolder_path, 'patches')
        target_image_dir = os.path.join(subfolder_path, str(dataset_label))
        create_dir(target_image_dir)

        logger.info(f"Starting to copy and rename image files to {target_image_dir}")
        for _, row in tqdm(mapping_df.iterrows(), total=mapping_df.shape[0], desc="Copying images"):
            old_name = row['old_name']
            new_name = row['new_name']
            src_image_path = os.path.join(source_patches_dir, f"{old_name}.png")
            dst_image_path = os.path.join(target_image_dir, f"{new_name}.png")
            try:
                shutil.copyfile(src_image_path, dst_image_path)
            except FileNotFoundError:
                logger.warning(f"Image file not found: {src_image_path}")
            except Exception as e:
                logger.error(f"Failed to copy image file: {src_image_path} -> {dst_image_path}, Error: {e}")

        logger.info(f"Completed copying and renaming images for dataset {dataset_label}.") if dataset_label is not None else ""

        df = df[['x', 'y', 'new_x', 'new_y', 'pixel_x', 'pixel_y', 'selected']]
        df.to_csv(f"{subfolder_path}/data/{dataset_label}_selection.tsv", sep='\t', index=False)

        filtered_gene_expression_data = adata_filtered.X.toarray()
        gene_expression_df = pd.DataFrame(filtered_gene_expression_data)
        gene_expression_df.index = index
        gene_expression_df.columns = adata_filtered.var_names
        np.save(f"{subfolder_path}/data/harmony_matrix.npy", gene_expression_df)
        gene_expression_df.to_parquet(f"{subfolder_path}/data/{dataset_label}_sub.parquet", engine='pyarrow', compression='snappy')
        gene_expression_df.to_csv( f"{subfolder_path}/data/{dataset_label}.tsv", sep='\t')

        barcodes = pd.DataFrame(adata_filtered.obs.index)
        barcodes.columns = ["barcodes"]
        barcodes.to_csv(f"{subfolder_path}/data/barcodes.tsv", index=False)


def create_comp_data_10x(input_dir, dataset_key, dataset_labels, spots_used_per_dataset, gene_type, analyzed_genes, mode, device,):
    valid_genes_consistent = True 
    for dataset_label in dataset_labels:
        print(dataset_label)
        subfolder_path = f'{input_dir}/{dataset_label}'
        create_dir(f"{subfolder_path}/10x_data")
        if dataset_key == "10x":
            adata = sc.read_visium(path=subfolder_path, count_file='filtered_feature_bc_matrix.h5')
            image_array = None  
        else:
            adata, image_array = create_adata_10x(input_dir, dataset_label)

        adata.var_names_make_unique()

        duplicated_vars = adata.var_names[adata.var_names.duplicated()]
        if duplicated_vars.any(): 
            raise ValueError(f"Duplicate variable names found in the current adata object: {duplicated_vars}")

        genes_not_in_adata = [gene for gene in analyzed_genes if gene not in adata.var_names]

        if genes_not_in_adata:
            logger.info(f"The following genes in dataset {dataset_label} were not found in adata: {genes_not_in_adata}")
            valid_genes_consistent = False  # Some genes are missing

        if valid_genes_consistent:
            valid_genes = list(analyzed_genes)
        else:
            valid_genes = [gene for gene in analyzed_genes if gene in adata.var_names]

        spot_used = spots_used_per_dataset[dataset_label]
        adata_filtered = adata[:, valid_genes]
        adata_filtered = adata_filtered[adata_filtered.obs_names.isin(spot_used), :]

        indices = [adata_filtered.obs_names.tolist().index(spot) for spot in spot_used if spot in adata_filtered.obs_names]
        adata_filtered = adata_filtered[indices, :] 
        save_as_10x(adata_filtered, dataset_key, f"{subfolder_path}/10x_data", image_array=image_array)


def create_adata_10x(data_path, dataset_label):
    expression_matrix = pd.read_csv(f"{data_path}/{dataset_label}/{dataset_label}.tsv.gz", sep="\t", index_col=0)
    spot_coordinates = pd.read_csv(f"{data_path}/{dataset_label}/{dataset_label}_selection.tsv.gz", sep="\t")

    spot_coordinates['spot'] = (spot_coordinates['x'].astype(str) + 'x' + spot_coordinates['y'].astype(str))

    common_spots = expression_matrix.index.intersection(spot_coordinates['spot'])
    expression_matrix = expression_matrix.loc[common_spots]
    spot_coordinates = (spot_coordinates.set_index('spot').loc[common_spots].reset_index())

    if 'selected' not in spot_coordinates.columns:
        spot_coordinates['selected'] = 1
    spot_coordinates_df = spot_coordinates[['index', 'selected', 'new_x', 'new_y', 'pixel_x', 'pixel_y']]
    spot_coordinates_df = spot_coordinates_df.rename(columns={'index': 'barcodes'})
    
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

    sparse_matrix = csr_matrix(expression_matrix.values)
    adata = anndata.AnnData(X=sparse_matrix, obs=pd.DataFrame(index=expression_matrix.index), var=pd.DataFrame(index=expression_matrix.columns))
    supp_info = spot_coordinates[['x', 'y', 'new_x', 'new_y']]
    adata.obsm['supp_info'] = supp_info.values
    adata.obsm['spatial'] = spot_coordinates[['pixel_x', 'pixel_y']].values
    adata.obs['neighbor_distance'] = spot_coordinates['neighbor_distance'].values
    adata.uns['spatial'] = image_data

    return adata, image_array


def save_as_10x(adata, dataset_key, output_path, image_array = None):
    os.makedirs(f"{output_path}/spatial", exist_ok=True)
    if image_array is not None:
        Image.fromarray(image_array).save(
            f"{output_path}/spatial/tissue_hires_image.png")
        Image.fromarray(image_array).save(
            f"{output_path}/spatial/tissue_lowres_image.png")
    else:
        print("Skipping image saving; assuming images already exist in the target directory.")
  
    scalefactors = adata.uns['spatial'][list(adata.uns['spatial'].keys())[0]]['scalefactors']
    with open(f"{output_path}/spatial/scalefactors_json.json", 'w') as f:
        json.dump(scalefactors, f)
    if dataset_key == "10x":
        df = pd.DataFrame({
            'x': adata.obs['array_row'].values,
            'y': adata.obs['array_col'].values,
            'new_x': adata.obs['array_row'].values,
            'new_y': adata.obs['array_col'].values,
            'pixel_x': adata.obsm['spatial'][:, 0],  
            'pixel_y': adata.obsm['spatial'][:, 1]   
        })

    else:
        df = pd.DataFrame({
            'x': adata.obsm['supp_info'][:, 0],
            'y': adata.obsm['supp_info'][:, 1],
            'new_x': adata.obsm['supp_info'][:, 2],
            'new_y': adata.obsm['supp_info'][:, 3],
            'pixel_x': adata.obsm['spatial'][:, 0], 
            'pixel_y': adata.obsm['spatial'][:, 1]  
        })
        df['x'] = df['x'].astype(int)
        df['y'] = df['y'].astype(int)
        generated_index = df['x'].astype(str) + 'x' + df['y'].astype(str)
        if list(generated_index) == list(adata.obs_names):
            print("Index content and order are exactly the same!")
        else:
            for i, (gen_idx, obs_idx) in enumerate(zip(generated_index, adata.obs_names)):
                if gen_idx != obs_idx:
                    print(f"Mismatch at position {i}: generated_index = {gen_idx}, but adata.obs_names = {obs_idx}")
                    break

    index = adata.obs_names  
    df['selected'] = 1  
    df['barcode'] = index
    df = df[['barcode', 'selected', 'x', 'y', 'pixel_x', 'pixel_y']]
    df.to_csv(f"{output_path}/spatial/tissue_positions_list.csv", index=False, header=False)
    adata.write_h5ad(f"{output_path}/filtered_feature_bc_matrix.h5")
