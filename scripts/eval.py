import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from mgcn import *
from utils import *
from gene2vec import load_gene_embeddings



device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
mode = "eval"

mgcn_gnn_type = "gat"
gene_gnn_type = "gcn"

dataset_key = "other"
species =  "human"
input_dir = r'./data/肾癌'
dataset_labels = ["GSM4284316", "GSM4284317", "GSM4284318", "GSM4284319", "GSM4284320", "GSM4284321", "GSM4284322", "GSM4284323", "GSM4284324", "GSM4284325", "GSM4284326", "GSM4284327"]
eval_label = "GSM4284319"
marker_genes = ''

gene_type = "highly_variable"  # "highly_expressed"
n_top_genes = 3000

gene_embeddings_path = './scripts/MultiDimGCN/biological_database/gene2vec_dim_200_iter_9_w2v.txt'
gene_embeddings = load_gene_embeddings(gene_embeddings_path)
gene_embeddings_keys = gene_embeddings.keys()

features_and_adjacencies_per_dataset, spots_used_per_dataset  = preprocess_data(input_dir, dataset_key, [eval_label], [eval_label], gene_type, n_top_genes, gene_embeddings_keys, marker_genes, mgcn_gnn_type, mode, device)
with open(f"{input_dir}/preprocess_output/4.shared_highly_variable_genes.txt", 'r') as f:
    shared_highly_variable_genes = [line.strip() for line in f]

raw_gene_expression_data, adata_hvgs = get_gene_expression(input_dir, dataset_key, [eval_label], spots_used_per_dataset, gene_type, shared_highly_variable_genes, mode, device)

img_features_per_dataset = check_image_feature(input_dir, dataset_key, [eval_label], [eval_label], spots_used_per_dataset, mode, device)

def load_model(input_dims, hidden_dims, output_dim, alpha, input_dir, device, gene_embeddings, mgcn_gnn_type, gene_gnn_type):

    model_gene_embedding = GNN(200, hidden_dims[0], output_dim, gene_gnn_type).to(device)
    model_mgcn = mGCN(input_dims, hidden_dims, output_dim, mgcn_gnn_type, alpha).to(device)
    
    try:
        checkpoint_path = f'{input_dir}/model_pth/best_model.pth'
        checkpoint = torch.load(checkpoint_path)
        model_mgcn.load_state_dict(checkpoint['model_mgcn'])
        model_gene_embedding.load_state_dict(checkpoint['model_gene_embedding'])
    
    except FileNotFoundError:
        print(f"模型文件 {checkpoint_path} 未找到")
        return None, None

    return model_mgcn, model_gene_embedding


input_dims = [19, 27, 2048]
hidden_dims = [256,512, 1024]
output_dim = 512
alpha = 0.9 # 0.75

model_mgcn, model_gene_embedding = load_model(input_dims, hidden_dims, output_dim, alpha, input_dir, device, gene_embeddings, mgcn_gnn_type, gene_gnn_type)

gene_embeddings_tensor = torch.tensor([gene_embeddings[gene] for gene in shared_highly_variable_genes], dtype=torch.float32).to(device)
gene_similarity_matrix_path =  f'{input_dir}/preprocess_output/5.{gene_type}_gene_similarity_matrix.csv'
gene_similarity_matrix = pd.read_csv(gene_similarity_matrix_path, index_col=0)
gene_similarity_matrix_tensor = process_gene_similarity_matrix(shared_highly_variable_genes, gene_similarity_matrix, device)
gene_similarity_matrix_tensor_dgl = convert_adj_matrix_to_dgl_graph(gene_similarity_matrix_tensor, gene_gnn_type, device)


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


import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def calculate_correlations(predicted_tensor, true_tensor, spots_used, intersection_genes, n_top_genes, input_dir):
    predicted_array = predicted_tensor.cpu().numpy()
    true_array = true_tensor.cpu().numpy()

    spots_used = list(spots_used)
    intersection_genes = list(intersection_genes)

    model_pth_dir = f"{input_dir}/model_pth"
    os.makedirs(model_pth_dir, exist_ok=True)

    predicted_df = pd.DataFrame(predicted_array, index=spots_used, columns=intersection_genes)
    true_df = pd.DataFrame(true_array, index=spots_used, columns=intersection_genes)

    predicted_df.to_csv(f"{model_pth_dir}/1.predicted_expression.csv")
    true_df.to_csv(f"{model_pth_dir}/2.true_df.csv")

    sample_id = os.path.basename(input_dir)

    if sample_id.startswith("train_eval_"):
        sample_id = sample_id.replace("train_eval_", "", 1)

    parent_dir = os.path.dirname(input_dir)
    selected_gene_file = f"{parent_dir}/top50_genes/{sample_id}.txt"
    selected_genes = []

    with open(selected_gene_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                genes = line.replace(",", " ").split()
                selected_genes.extend(genes)

    selected_genes = list(dict.fromkeys(selected_genes))

    available_genes = set(intersection_genes)
    missing_genes = [gene for gene in selected_genes if gene not in available_genes]
    valid_selected_genes = [gene for gene in selected_genes if gene in available_genes]

    if len(valid_selected_genes) == 0:
        raise ValueError("txt 中的基因和 intersection_genes 完全没有交集，无法计算相关性。")

    if len(missing_genes) > 0:
        print(f"[警告] 有 {len(missing_genes)} 个基因不在 intersection_genes 中:")
        print(missing_genes)

    correlations = []

    for gene in valid_selected_genes:
        pred_gene = predicted_df[gene]
        true_gene = true_df[gene]

        if pred_gene.nunique(dropna=True) <= 1 or true_gene.nunique(dropna=True) <= 1:
            corr = np.nan
        else:
            corr, _ = pearsonr(pred_gene, true_gene)

        correlations.append(corr)

    correlation_df = pd.DataFrame({
        "Gene": valid_selected_genes,
        "Correlation": correlations
    })

    output_file = f"{model_pth_dir}/3.gene_correlations.csv"
    correlation_df.to_csv(output_file, index=False)

    mean_correlation = correlation_df["Correlation"].mean()
    print(f"平均相关性: {mean_correlation:.4f}")

    if len(missing_genes) > 0:
        print(f"未匹配到的基因数量: {len(missing_genes)}")
        print(missing_genes)

    return valid_selected_genes, correlation_df

n_top_genes = 50
top_n_genes,correlation_df = calculate_correlations(predicted_expression, gene_expression_data, spots_used_per_dataset[eval_label], shared_highly_variable_genes, n_top_genes, input_dir)

# 计算各自 Min-Max 归一化后的 MSE
eps = 1e-8

pred_min = predicted_expression.min(dim=0, keepdim=True).values
pred_max = predicted_expression.max(dim=0, keepdim=True).values
pred_range = (pred_max - pred_min).clamp_min(eps)

true_min = gene_expression_data.min(dim=0, keepdim=True).values
true_max = gene_expression_data.max(dim=0, keepdim=True).values
true_range = (true_max - true_min).clamp_min(eps)

predicted_expression_norm = (predicted_expression - pred_min) / pred_range
gene_expression_data_norm = (gene_expression_data - true_min) / true_range

print(F.mse_loss(predicted_expression_norm, gene_expression_data_norm).item())



