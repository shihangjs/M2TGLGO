###  Step 0: Patch Cropping (Based on Spatial Transcriptomics Spots)

> This is the **initial step** of the pipeline. Before nuclear segmentation, you must crop image patches around spatial transcriptomics **spot locations** from high-resolution histology images.  
> Each sample (e.g., `A1`, `B1`, etc.) contains its own image and corresponding spot coordinate files.

---

####  Input Format

Each sample directory (e.g., `A1`, `B1`, etc.) should contain the following files:

| File Name                  | Description                                      |
|----------------------------|--------------------------------------------------|
| `*.jpg`                   | Histology tissue image for the sample            |
| `*.tsv.gz`                | Spot coordinate file (with x/y positions)        |
| `*_selection.tsv.gz`      | *(Optional)* A filtered list of selected spots   |

These files will be used to generate image patches centered on each spatial spot.

---

#### Example Directory Structure

```
data/
├── A1/
│   ├── A1.jpg
│   ├── A1.tsv.gz
│   └── A1_selection.tsv.gz
├── A2/
│   ├── A2.jpg
│   ├── A2.tsv.gz
│   └── A2_selection.tsv.gz
├── ...
├── B1/
│   ├── B1.jpg
│   ├── B1.tsv.gz
│   └── B1_selection.tsv.gz
```

---

####  Output

Cropped image patches will be saved to:

```
/data/{SAMPLE_ID}/patches/
```

Each patch corresponds to one spatial spot, and will be used in downstream nuclear segmentation.

---


#### Data Loading Guide: Support for Multiple Data Types

This pipeline supports **two types** of spatial transcriptomics datasets:

| Dataset Type | Source Format             | Loading Method                           |
|--------------|---------------------------|------------------------------------------|
| `10x`        | 10x Visium standard format | Loaded via `scanpy.read_visium()`        |
| `other`      | Custom (tsv/image/coords)  | Loaded using a custom `create_adata()` function |

---

#### Data Loading Logic

The script will automatically choose the appropriate loading strategy based on the `dataset_key`:

```python
if dataset_key == "10x":
    adata = sc.read_visium(
        path=subfolder_path,
        count_file='filtered_feature_bc_matrix.h5'
    )
else:
    adata, _ = create_adata(input_dir, dataset_label)
```

---

#### `create_adata()` Function: Build AnnData for Custom Datasets

This function builds a standard `AnnData` object from a raw expression `.tsv` file, a high-resolution `.jpg` image, and a coordinate file.  
You may need to adjust the file paths or column names based on your specific data.

---

#### `create_adata()` Function: Build AnnData for Custom Datasets Required Files per Sample

Each dataset directory should contain:

| File Name                         | Description                                 |
|----------------------------------|---------------------------------------------|
| `{dataset_label}.tsv.gz`         | Spot × Gene expression matrix               |
| `{dataset_label}_selection.tsv.gz` | Spot coordinates (including pixel_x/y)     |
| `{dataset_label}.jpg`            | High-resolution tissue image                |

---

#### `create_adata()` Function: Build AnnData for Custom Datasets Example Code for `create_adata()`

```python
def create_adata(data_path, dataset_label):
    import pandas as pd
    import numpy as np
    from PIL import Image
    from scipy.spatial import cKDTree
    import anndata

    # 1. Load expression matrix
    expression_matrix = pd.read_csv(
        f"{data_path}/{dataset_label}/{dataset_label}.tsv.gz",
        sep="\t", index_col=0
    )

    # 2. Load spot coordinates
    spot_coordinates = pd.read_csv(
        f"{data_path}/{dataset_label}/{dataset_label}_selection.tsv.gz",
        sep="\t"
    )

    # 3. Generate spot IDs and match with expression matrix
    spot_coordinates['spot'] = spot_coordinates['x'].astype(str) + 'x' + spot_coordinates['y'].astype(str)
    common_spots = expression_matrix.index.intersection(spot_coordinates['spot'])
    expression_matrix = expression_matrix.loc[common_spots]
    spot_coordinates = spot_coordinates.set_index('spot').loc[common_spots].reset_index()

    # 4. Compute neighbor distance (for patch size estimation)
    coords = spot_coordinates[['pixel_x', 'pixel_y']].values
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=5)
    min_distances = distances[:, 1:].min(axis=1)
    spot_coordinates['neighbor_distance'] = min_distances

    # 5. Load tissue image
    image_path = f"{data_path}/{dataset_label}/{dataset_label}.jpg"
    image = Image.open(image_path)
    image_array = np.array(image)

    # 6. Construct AnnData
    adata = anndata.AnnData(X=expression_matrix)

    # 7. Add metadata
    adata.obsm['spatial'] = spot_coordinates[['pixel_x', 'pixel_y']].values
    adata.obsm['supp_info'] = spot_coordinates[['x', 'y', 'new_x', 'new_y']].values
    adata.obs['neighbor_distance'] = spot_coordinates['neighbor_distance'].values
    adata.uns['spatial'] = {
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

    return adata, image_array
```

---

#### `create_adata()` Function: Build AnnData for Custom Datasets Notes

- The `neighbor_distance` is used to approximate the full-resolution spot size.
- The `adata.uns['spatial']` field mimics 10x Visium's data structure, which helps with downstream compatibility.
- Modify this function if your data format differs.


###  Step 1: Nuclear Segmentation (Using Hover-Net)

> This step uses a **pretrained Hover-Net model** to perform **nucleus segmentation** on each image patch.  
> The output is a per-spot segmentation mask and cell type annotation, which will be used in the next step (feature extraction).


#### Hover-Net Segmentation Script (`run_hovernet.sh`)

Here is an example shell script to segment nuclei for multiple datasets using Hover-Net:

```bash
#!/bin/bash

datasets=(
    A1 A2 A3 A4 A5 A6
    B1 B2 B3 B4 B5 B6
)

for dataset in "${datasets[@]}"; do
    python3 run_infer.py \
        --gpu='3' \
        --nr_types=6 \
        --type_info_path=type_info.json \
        --batch_size=32 \
        --model_mode=fast \
        --model_path=./pretrained/hovernet_fast_pannuke_type_tf2pytorch.tar \
        --nr_inference_workers=8 \
        --nr_post_proc_workers=16 \
        tile \
        --input_dir="/data/${dataset}/patches" \
        --output_dir="/data/${dataset}/segment" \
        --mem_usage=0.1 \
        --draw_dot \
        --save_qupath
done
```

You may save this script as `run_hovernet.sh` and execute:

```bash
bash run_hovernet.sh
```

---


###  Step 2: Feature Extraction (from Segmented Nuclei)

> After segmentation, this step extracts **patch-level features** from each patch using both image and segmentation data.  
The script [`patch_cell_feature_extract_.py`](preprocess/patch_cell_feature_extract_.py) processes all required files.

---

####  Required Folder Structure

Each sample (e.g., `A1`, `B1`) must include the following subdirectories:

```
data/
└── B1/
    ├── patches/                 # Cropped image patches (from Step 0)
    ├── segment/
    │   ├── mat/                 # Hover-Net segmentation outputs (.mat or .npy files)
    │   └── json/                # JSON files with nucleus contours / cell type info
    └── feature/                 # [Output] Extracted feature CSVs will be saved here
```

---

####  Example Feature Extraction Loop

If you're processing multiple samples, you can run a loop like this:

```python
files = ["A1", "A2", "A3", "B1", "B2", "B3"]
input_dir = "data"
segment_tool = "hovernet"

for file in files:
    print(file)
    dataset_label = file
    mat_dir = f"{input_dir}/{dataset_label}/segment/mat"
    output_dir = f"{input_dir}/{dataset_label}/feature"
    patches_dir = f"{input_dir}/{dataset_label}/patches"
    json_dir = f"{input_dir}/{dataset_label}/segment/json"

    process_files(
        mat_dir=mat_dir,
        output_dir=output_dir,
        dataset_label=dataset_label,
        patches_dir=patches_dir,
        json_dir=json_dir,
        segment_tool=segment_tool
    )
```

Make sure `process_files()` is imported or defined in your script.

---

###  Step 3: Model Training and Evaluation (Multi-Modal Graph Neural Network)

> In this step, M2TGLO is trained to predict gene expression levels from cell-level, patch-level, and image-level features.  
> The model also incorporates **gene embeddings** (from gene2vec) and **GO-based gene similarity graphs** to regularize prediction.

---

####  Input Requirements

Your training input directory should be organized as follows:

```
{input_dir}/
├── A1/ ~ B6/
│   ├── patches/                 # From Step 0
│   ├── segment/                # From Step 1
│   ├── feature/                # From Step 2
│
│
├── biological_database/
│   ├── gene2vec_dim_200_iter_9_w2v.txt
│   └── go-basic.obo
```

---

### Step 3.1: Configure Training Settings

Set up paths, training mode, gene filtering strategy, and evaluation split.

```python

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# GNN architecture types
mgcn_gnn_type = "gat"        # For mGCN (multi-branch graph for spatial features)
gene_gnn_type = "gat"        # For gene-level graph (based on GO similarity)

# Dataset configuration
dataset_key = "other"        # Key to distinguish dataset
species = "human"            # Used for gene ontology lookup

# Directory containing processed input data
input_dir = "/data/"

# List of all available datasets
dataset_labels = [
    "A1", "A2", "A3", "A4", "A5", "A6",
    "B1", "B2", "B3", "B4", "B5", "B6"
]

# Leave-one-out evaluation setup
eval_label = "A1"  # Dataset held out for evaluation
train_dataset_labels = list(set(dataset_labels) - set([eval_label]))

# Optional: specify marker genes to include (default is empty)
marker_genes = ''

# Directory to save experiment outputs
create_dir(f"{input_dir}/experiments")
train_folder = f"{input_dir}/experiments/{eval_label}"
create_dir(train_folder)

# Mode can be "train" or "eval"
mode = "train"

# Gene selection strategy
gene_type = "highly_variable"   
n_top_genes = 3000              # Number of top genes to retain for modeling
```

---

### Step 3.2: Load or Generate Training Data

> This step **automatically checks for existing processed data** (e.g., `.pt`, `.csv`, `.npy`) and loads them if available.  
> If not found, it will **run the corresponding preprocessing logic** to generate the necessary files.

---

#### Data Preparation Code

```python
# Load or preprocess features & graph structure
features_and_adjacencies_per_dataset, analyzed_genes, spots_used_per_dataset = check_preprocess_data(
    input_dir=input_dir,
    dataset_key=dataset_key,
    dataset_labels=dataset_labels,
    train_dataset_labels=train_dataset_labels,
    gene_type=gene_type,
    n_top_genes=n_top_genes,
    gene_embeddings_keys=gene_embeddings.keys(),
    marker_genes=marker_genes,
    mgcn_gnn_type=mgcn_gnn_type,
    mode=mode,
    device=device
)

# Load or compute raw gene expression matrix (per spot)
raw_gene_expression_data = check_gene_expression(
    input_dir, dataset_key, train_dataset_labels, spots_used_per_dataset,
    gene_type, analyzed_genes, mode, device
)

# Load or compute image-level features per patch (e.g., CNN-based vectors)
img_features_per_dataset = check_image_feature(
    input_dir, dataset_key, dataset_labels, train_dataset_labels,
    spots_used_per_dataset, gene_type, mode, device
)

# Load or build gene similarity matrix from GO ontology
GODag_path = "sh/scripts/MultiDimGCN/biological_database/go-basic.obo"
gene_similarity_matrix = check_gene_similarity_matrix(
    input_dir, species, gene_type, analyzed_genes, GODag_path
)
```

---

##### ⚠️ Note on GO-based Gene Similarity Computation

- The function `check_gene_similarity_matrix()` may take significant time on the **first run**, as it computes semantic similarity between genes using the GO DAG.
- You can set the number of parallel CPU **cores/threads** used for this step to accelerate the process.
- If the process crashes or is interrupted (e.g., due to memory issues or I/O), **do not worry**:
  - The code automatically saves intermediate results (e.g., partially completed similarity matrices).
  - You can **simply re-run the script**, and it will resume from where it left off instead of starting over.


### Step 3.3: Train the Multi-Modal GCN Model

Once all data is loaded and preprocessed, you can start model training using the following function call:

```python
train(
    input_dir=train_folder,
    dataset_labels=train_dataset_labels,
    features_and_adjacencies_per_dataset=features_and_adjacencies_per_dataset,
    img_features_per_dataset=img_features_per_dataset,
    raw_gene_expression_data=raw_gene_expression_data,
    gene_embeddings=gene_embeddings,
    gene_similarity_matrix=gene_similarity_matrix,
    mgcn_gnn_type=mgcn_gnn_type,
    gene_gnn_type=gene_gnn_type,
    mode=mode,
    device=device
)
```


### Step 3.4: Evaluate on Held-Out Dataset

After training is complete, you can evaluate the model on the **held-out dataset** (specified by `eval_label`) by switching to evaluation mode:
```python
mode = "eval"
```

```python
model_mgcn.eval()
model_gene_embedding.eval()

with torch.no_grad():
    updated_gene_embeddings = model_gene_embedding(gene_similarity_matrix_tensor_dgl, gene_embeddings_tensor)

    data = features_and_adjacencies_per_dataset[eval_label]
    gene_expression_data = raw_gene_expression_data[eval_label]
    adj_matrix = convert_adj_matrix_to_dgl_graph(data['cumulative_adj_matrix'], mgcn_gnn_type, device)

    node_features_list = [
        data['filtered_cell_level_features_tensor'],
        data['filtered_patch_level_features_tensor'],
        img_features_per_dataset[eval_label]
    ]

    node_embeddings = model_mgcn(node_features_list, adj_matrix)

    predicted_expression = torch.matmul(node_embeddings, updated_gene_embeddings.T)
```

This produces a **predicted expression matrix** for the held-out dataset.

---

#### Output

You can optionally save the predicted expression as `.csv` or `.npy` for downstream comparison with the ground truth:

```python
# Optional save
torch.save(predicted_expression.cpu(), f"{train_folder}/predicted_expression.pt")
```
