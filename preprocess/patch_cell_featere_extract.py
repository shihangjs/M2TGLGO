import os
import json
import numpy as np
import pandas as pd
import scipy.io as scio
from tqdm import tqdm
from skimage.measure import label, regionprops, perimeter
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.morphology import convex_hull_image, skeletonize
from skimage.color import rgb2gray
from scipy.spatial import Delaunay, Voronoi
from scipy.spatial.distance import cdist
from PIL import Image
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

def load_mat_file(file_path, segment_tool):
    mat = scio.loadmat(file_path)
    if segment_tool == "matlab":
        data = mat.get('segmentedImage', None)
    else:
        data = mat.get('inst_map', None)
    if data is None:
        raise ValueError(f"Failed to load segmentation data from {file_path}.")
    data = data.astype(np.uint8)
    return data

def load_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def get_texture_features(region_mask):
    distances = [1]  
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'] 
    levels = np.max(region_mask) + 1 
    glcm = graycomatrix(region_mask, distances, angles, levels=levels, symmetric=True, normed=True)

    texture_features = {prop: np.mean(graycoprops(glcm, prop)) for prop in props}

    epsilon = np.finfo(float).eps 
    glcm += epsilon 
    entropy = -np.sum(glcm * np.log2(glcm))
    texture_features['entropy'] = entropy

    mean = np.mean(glcm)
    texture_features['mean'] = mean

    variance = np.var(glcm)  
    texture_features['variance'] = variance

    glcm_sum = np.sum(glcm, axis=(0, 1))
    glcm_diff = np.abs(glcm - np.roll(glcm, 1, axis=(0, 1))) 

    texture_features['sum_average'] = np.mean(glcm_sum)
    texture_features['sum_variance'] = np.var(glcm_sum)
    texture_features['sum_entropy'] = -np.sum(glcm_sum * np.log2(glcm_sum + epsilon))

    texture_features['difference_average'] = np.mean(glcm_diff)
    texture_features['difference_variance'] = np.var(glcm_diff)
    texture_features['difference_entropy'] = -np.sum(glcm_diff * np.log2(glcm_diff + epsilon))

    px = np.sum(glcm, axis=0)
    py = np.sum(glcm, axis=1)
    hx = -np.sum(px * np.log2(px + epsilon))
    hy = -np.sum(py * np.log2(py + epsilon))
    hxy1 = -np.sum(glcm * np.log2(glcm))
    px_py = px * py
    hxy2 = -np.sum(px_py * np.log2(px_py + epsilon))

    denominator = max(hx, hy)
    if denominator == 0:
        texture_features['information_measure_corr1'] = 0
    else:
        texture_features['information_measure_corr1'] = (hxy1 - hxy2) / denominator


    exponent = -2 * (hxy2 - hxy1)
    value = 1 - np.exp(exponent)
    value = np.clip(value, a_min=0, a_max=1) 
    texture_features['information_measure_corr2'] = np.sqrt(value)

    lbp = local_binary_pattern(region_mask, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= lbp_hist.sum() + epsilon
    texture_features.update({f'LBP_{i}': v for i, v in enumerate(lbp_hist)})

    return texture_features


def extract_cell_features(label_image, image_gray, centroids):
    features = []

    regions = regionprops(label_image)
    labels = [region.label for region in regions]
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    if len(centroids) != len(labels):
        print(f"Warning: Number of labels ({len(labels)}) does not match number of centroids ({len(centroids)}).")

    if len(centroids) >= 3:
        tri = Delaunay(centroids)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    edges.add(edge)

        neighbor_count = {i: 0 for i in range(len(centroids))}
        for edge in edges:
            neighbor_count[edge[0]] += 1
            neighbor_count[edge[1]] += 1

        distances = cdist(centroids, centroids)
        np.fill_diagonal(distances, np.inf)
        nearest_neighbor_distances = np.min(distances, axis=1)
        furthest_neighbor_distances = np.max(distances, axis=1)  
        vor = Voronoi(centroids)
        voronoi_areas = []
        for region_index in vor.point_region:
            region = vor.regions[region_index]
            if -1 in region or len(region) == 0:
                voronoi_areas.append(0)
                continue
            polygon = vor.vertices[region]
            area = 0.5 * np.abs(
                np.dot(polygon[:, 0], np.roll(polygon[:, 1], 1)) -
                np.dot(polygon[:, 1], np.roll(polygon[:, 0], 1)))
            voronoi_areas.append(area)

        average_neighbor_distances = []
        furthest_neighbor_distances_neighbor = []
        for i in range(len(centroids)):
            neighbor_indices = [j for j in range(len(centroids)) if (i, j) in edges or (j, i) in edges]
            neighbor_distances = [np.linalg.norm(centroids[i] - centroids[j]) for j in neighbor_indices]
            if neighbor_distances:
                average_neighbor_distances.append(np.mean(neighbor_distances))
                furthest_neighbor_distances_neighbor.append(np.max(neighbor_distances))
            else:
                average_neighbor_distances.append(0)
                furthest_neighbor_distances_neighbor.append(0)
    else:
        neighbor_count = {i: 0 for i in range(len(centroids))}
        nearest_neighbor_distances = [0] * len(centroids)
        furthest_neighbor_distances = [0] * len(centroids)
        average_neighbor_distances = [0] * len(centroids)
        furthest_neighbor_distances_neighbor = [0] * len(centroids)
        voronoi_areas = [0] * len(centroids)

    for region in regions:
        cell_mask = (label_image == region.label).astype(np.uint8)
        cell_convex_hull = convex_hull_image(cell_mask)
        convex_area = np.sum(cell_convex_hull)
        convex_perimeter = perimeter(cell_convex_hull)

        cell_features = {
            'Label': region.label,
            'Area': region.area,
            'Perimeter': region.perimeter,
            'Eccentricity': region.eccentricity,
            'MajorAxisLength': region.major_axis_length,
            'MinorAxisLength': region.minor_axis_length,
            'Solidity': region.solidity,
            'Extent': region.extent,
            'Orientation': region.orientation,
            'Circularity': (4 * np.pi * region.area / (region.perimeter ** 2)) if region.perimeter > 0 else 0,
            'AspectRatio': (region.major_axis_length / region.minor_axis_length) if region.minor_axis_length > 0 else 0,
            'ConvexArea': convex_area,
            'Convexity': (region.area / convex_area) if convex_area > 0 else 0,
            'EulerNumber': region.euler_number,
            'SkeletonLength': np.sum(skeletonize(cell_mask)),
            'ConvexPerimeter': convex_perimeter
        }

        texture_features = get_texture_features(cell_mask)
        cell_features.update(texture_features)

        label_id = label_to_index.get(region.label, None)
        if label_id is not None and label_id < len(centroids):
            cell_features.update({
                'NeighborCount': neighbor_count.get(label_id, 0),
                'NearestNeighborDistance': nearest_neighbor_distances[label_id],
                'AverageNeighborDistance': average_neighbor_distances[label_id],
                'FurthestNeighborDistance': furthest_neighbor_distances_neighbor[label_id],
                'VoronoiArea': voronoi_areas[label_id]
            })
        else:
            cell_features.update({
                'NeighborCount': 0,
                'NearestNeighborDistance': 0,
                'AverageNeighborDistance': 0,
                'FurthestNeighborDistance': 0,
                'VoronoiArea': 0
            })
        features.append(cell_features)

    return pd.DataFrame(features)


def process_files(mat_dir, output_dir, dataset_label, patches_dir, json_dir, segment_tool):

    os.makedirs(output_dir, exist_ok=True)
    cell_feature_dir = os.path.join(output_dir, 'cell_level_features')
    patch_feature_dir = os.path.join(output_dir, 'patch_level_features')
    os.makedirs(cell_feature_dir, exist_ok=True)
    os.makedirs(patch_feature_dir, exist_ok=True)

    summary = []
    all_cell_features = []
    all_patch_features = []

    mat_files = [f for f in os.listdir(mat_dir) if f.endswith('.mat')]

    for file_name in tqdm(mat_files, desc=f"Processing files in {dataset_label}"):
        mat_file_path = os.path.join(mat_dir, file_name)
        try:
            data = load_mat_file(mat_file_path, segment_tool)
            label_image = label(data)
        except Exception as e:
            print(f"Error loading MAT file {mat_file_path}: {e}")
            continue

        base_file_name = os.path.splitext(file_name)[0]
        json_file_path = os.path.join(json_dir, base_file_name + '.json')
        try:
            json_data = load_json_file(json_file_path)
        except Exception as e:
            print(f"Error loading JSON file {json_file_path}: {e}")
            continue

        image_file_name = base_file_name + '.png'
        image_path = os.path.join(patches_dir, image_file_name)
        try:
            image = Image.open(image_path)
            image_rgb = np.array(image)
            image_gray = rgb2gray(image_rgb)
        except Exception as e:
            print(f"Error loading image file {image_path}: {e}")
            continue

        if json_data.get('nuc'):
            num_cells_mat = len(np.unique(label_image)) - 1  
            num_cells_json = len(json_data['nuc'])


            centroids = []
            for i in range(1, num_cells_json + 1):
                cell_info = json_data['nuc'].get(str(i), None)
                if cell_info is not None:
                    centroids.append(cell_info['centroid'])
                else:
                    centroids.append([0, 0]) 

            centroids = np.array(centroids)

            features = extract_cell_features(label_image, image_gray, centroids)

            columns_to_extract_cell = [
                'Label', 'Area', 'Perimeter', 'Eccentricity',
                'MajorAxisLength', 'MinorAxisLength', 'Solidity', 'Extent',
                'Orientation', 'Circularity', 'AspectRatio', 'ConvexArea',
                'Convexity', 'EulerNumber', 'SkeletonLength', 'ConvexPerimeter',
                'NeighborCount', 'NearestNeighborDistance',
                'AverageNeighborDistance', 'FurthestNeighborDistance',
                'VoronoiArea'
            ]

            cell_features = features[columns_to_extract_cell]
            cell_features_path = os.path.join(cell_feature_dir, f"{base_file_name}.csv")
            cell_features.to_csv(cell_features_path, index=False)

            cell_features_mean = cell_features.iloc[:, 1:].mean().to_frame().T
            cell_features_mean['dataset_label'] = dataset_label
            cell_features_mean['file_name'] = base_file_name
            all_cell_features.append(cell_features_mean)

            columns_to_extract_patch = features.columns[features.columns.isin(['Label']) | features.columns.str.startswith('LBP_') | features.columns.str.contains('contrast|dissimilarity|homogeneity|energy|correlation|ASM|entropy|mean|variance|sum|difference|information_measure')]

            texture_features = features[columns_to_extract_patch]
            patch_features_path = os.path.join(patch_feature_dir, f"{base_file_name}.csv")
            texture_features.to_csv(patch_features_path, index=False)

            texture_features_mean = texture_features.iloc[:, 1:].mean().to_frame().T
            texture_features_mean['dataset_label'] = dataset_label
            texture_features_mean['file_name'] = base_file_name
            all_patch_features.append(texture_features_mean)

            summary.append([dataset_label, base_file_name, num_cells_json])
        else:
            summary.append([dataset_label, base_file_name, 0])

    summary_df = pd.DataFrame(summary, columns=['dataset_label', 'file_name', 'cell_count'])
    summary_path = os.path.join(output_dir, 'cell_count.csv')
    summary_df.to_csv(summary_path, index=False)

    if all_cell_features:
        all_cell_features_df = pd.concat(all_cell_features, ignore_index=True)
        all_cell_features_path = os.path.join(output_dir, 'cell_level_features.csv')
        all_cell_features_df.to_csv(all_cell_features_path, index=False)

    if all_patch_features:
        all_patch_features_df = pd.concat(all_patch_features, ignore_index=True)
        all_patch_features_path = os.path.join(output_dir, 'patch_level_features.csv')
        all_patch_features_df.to_csv(all_patch_features_path, index=False)


def main(input_dir, segment_tool):

    exclude_dirs = {'training_experiments', 'model_pth', 'preprocess_output'}
    datasets = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f)) and f not in exclude_dirs]
    for dataset_label in datasets:
        mat_dir = os.path.join(input_dir, dataset_label, "segment", "mat")
        output_dir = os.path.join(input_dir, dataset_label, "feature")
        patches_dir = os.path.join(input_dir, dataset_label, "patches")
        json_dir = os.path.join(input_dir, dataset_label, "segment", "json")
        process_files(mat_dir, output_dir, dataset_label, patches_dir, json_dir, segment_tool)

if __name__ == "__main__":
    main()

