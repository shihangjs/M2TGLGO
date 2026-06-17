import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv, GATConv
from typing import List


class GNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, gnn_type: str):
        super(GNN, self).__init__()
        self.gnn_type = gnn_type.lower()

        if self.gnn_type == 'gcn':
            self.gcn1 = GraphConv(input_dim, output_dim)
        elif self.gnn_type == 'gat':
            num_heads = 1
            self.gat1 = GATConv(input_dim, output_dim, num_heads=num_heads, feat_drop=0.5)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

    def forward(self, graph: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        if self.gnn_type == 'gcn':
            x = F.relu(self.gcn1(graph, features))
        elif self.gnn_type == 'gat':
            x = F.relu(self.gat1(graph, features))
            x = x.view(x.shape[0], -1)
        return x


class ProjectionMatrix(nn.Module):
    def __init__(self, input_dims: List[int], output_dim: int):
        super(ProjectionMatrix, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(in_dim, output_dim) for in_dim in input_dims])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        return [F.relu(linear(feat)) for linear, feat in zip(self.linears, features)]


class Attention(nn.Module):
    def __init__(self, hidden_dim: int, num_modalities: int):
        super(Attention, self).__init__()
        
        self.transform_matrices = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_modalities)])
        for linear in self.transform_matrices:
            nn.init.xavier_uniform_(linear.weight)

    def forward(self, projected_features: List[torch.Tensor]) -> List[torch.Tensor]:
        D = len(projected_features)
        hidden_dim = projected_features[0].shape[1]
        device = projected_features[0].device
        transformed_features = [linear(feat) for linear, feat in zip(self.transform_matrices, projected_features)]
        attention_scores = torch.zeros(D, D, device=device)
        for i in range(D):
            for j in range(D):
                norm_i = F.normalize(transformed_features[i], p=2, dim=1)
                norm_j = F.normalize(transformed_features[j], p=2, dim=1)
                score_matrix = norm_i @ norm_j.T
                # score_ij = score_matrix.mean()
                # attention_scores[i, j] = score_ij
                diag_elements = torch.diagonal(score_matrix)
                attention_scores[i, j] = diag_elements.mean()
        attention_weights = F.softmax(attention_scores, dim=1)
        
        aggregated_features = []
        for d in range(D):
            weighted_sum = torch.zeros_like(projected_features[0])
            for g in range(D):
                weight = attention_weights[d, g]
                weighted_sum += weight * projected_features[g]
            aggregated_features.append(weighted_sum)

        return aggregated_features


class mGCNLayer(nn.Module):
    def __init__(self, input_dims: List[int], hidden_dims: List[int], output_dim: int, gnn_type: str, alpha: float):
        super(mGCNLayer, self).__init__()
        self.alpha = alpha
        self.projection = ProjectionMatrix(input_dims, hidden_dims)
        self.gnns = nn.ModuleList([GNN(hidden_dims, output_dim, gnn_type) for _ in input_dims])
        self.attention = Attention(hidden_dims, 3)
        self.linear_transform = nn.Linear(hidden_dims, output_dim)

    def forward(self, features: List[torch.Tensor], graph: dgl.DGLGraph) -> List[torch.Tensor]:
        projected_features = self.projection(features)
        intra_agg_features = [gnn(graph, feat) for gnn, feat in zip(self.gnns, projected_features)]
        inter_agg_features = self.attention(projected_features)
        inter_agg_features = [self.linear_transform(inter) for inter in inter_agg_features]
        combined_features = [self.alpha * intra + (1 - self.alpha) * inter for intra, inter in zip(intra_agg_features, inter_agg_features)]

        return combined_features


class ModalityAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_modalities: int):
        super(ModalityAttention, self).__init__()
        self.num_modalities = num_modalities
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.attention_fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.3)

        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.constant_(self.projection.bias, 0)
        nn.init.xavier_uniform_(self.attention_fc.weight)
        nn.init.constant_(self.attention_fc.bias, 0)

    def forward(self, modality_features: List[torch.Tensor]) -> torch.Tensor:
        projected_features = []
        attention_scores = []

        for feat in modality_features:
            proj_feat = F.tanh(self.projection(feat))
            projected_features.append(proj_feat)
            score = self.attention_fc(proj_feat)
            attention_scores.append(score)

        attention_scores = torch.stack(attention_scores, dim=1)
        attention_weights = F.softmax(attention_scores, dim=1)
        projected_features = torch.stack(projected_features, dim=1)
        fused_feature = torch.sum(attention_weights * projected_features, dim=1)  
        fused_feature = self.dropout(fused_feature)

        return fused_feature


class mGCN(nn.Module):
    def __init__(self, input_dims: List[int], hidden_dims: List[int], output_dim: int, gnn_type: str, alpha: float = 0.5):
        super(mGCN, self).__init__()
        self.layer1 = mGCNLayer(input_dims, hidden_dims[0], hidden_dims[1], gnn_type, alpha)
        self.layer2 = mGCNLayer([hidden_dims[1]] * len(input_dims), hidden_dims[1], hidden_dims[0], gnn_type, alpha)
        self.modality_attention = ModalityAttention(hidden_dims[0], hidden_dims[0], len(input_dims))
        self.fc = nn.Linear(hidden_dims[0], output_dim)

    def forward(self, features: List[torch.Tensor], graph: dgl.DGLGraph) -> torch.Tensor:
        features = self.layer1(features, graph)
        features = self.layer2(features, graph)
        fused_feature = self.modality_attention(features)  
        output = self.fc(fused_feature)  
        return output

def adjacency_reconstruction_loss(adj_3hop, embeddings, delta=0.1, alpha=0.01):
    recon_adj = F.relu(torch.mm(embeddings, embeddings.t()))
    loss = F.mse_loss(recon_adj, adj_3hop)
    return loss

def compute_quartiles_optimized(similarities: torch.Tensor) -> torch.Tensor:
    mask = similarities > 0  # [num_nodes, num_neighbors]
    masked_similarities = similarities.masked_fill(~mask, float('-inf'))  # [num_nodes, num_neighbors]
    sorted_similarities, _ = torch.sort( masked_similarities, dim=1, descending=True)  # [num_nodes, num_neighbors]

    counts = mask.sum(dim=1).unsqueeze(1)  # [num_nodes, 1]

    quartiles = torch.tensor([0.25, 0.5, 0.75, 1.0], device=similarities.device).view(1, -1)  # [1, 4]
    quartile_indices = (counts.float() * quartiles).ceil().long() - 1  # [1, 4]
    quartile_indices = quartile_indices.clamp(min=0, max=similarities.size(1) - 1)  
    quartiles_values = sorted_similarities.gather(1, quartile_indices)  # [num_nodes, 4]
    quartiles_values = quartiles_values.masked_fill(quartiles_values == float('-inf'), 0.0)
    sorted_quartiles_values, _ = torch.sort(quartiles_values, dim=1, descending=False)

    return sorted_quartiles_values


def node_similarity_loss(adj_1hop: torch.Tensor, adj_2hop: torch.Tensor, adj_3hop: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())  # [num_nodes, num_nodes]

    similarity_1hop = similarity_matrix * adj_1hop  # [num_nodes, num_nodes]
    similarity_2hop = similarity_matrix * adj_2hop  # [num_nodes, num_nodes]
    similarity_3hop = similarity_matrix * adj_3hop  # [num_nodes, num_nodes]

    # similarity_1hop.fill_diagonal_(0)
    # similarity_2hop.fill_diagonal_(0)
    # similarity_3hop.fill_diagonal_(0)

    q1_1hop = compute_quartiles_optimized(similarity_1hop)  # [num_nodes, 4]
    q1_2hop = compute_quartiles_optimized(similarity_2hop)  # [num_nodes, 4]
    q1_3hop = compute_quartiles_optimized(similarity_3hop)  # [num_nodes, 4]

    mean_q1_1hop = q1_1hop.mean(dim=0)  # [4]
    mean_q1_2hop = q1_2hop.mean(dim=0)  # [4]
    mean_q1_3hop = q1_3hop.mean(dim=0)  # [4]

    diff_1_3 = mean_q1_1hop - mean_q1_3hop  # [4]
    diff_2_3 = mean_q1_2hop - mean_q1_3hop  # [4]

    margin = diff_2_3 - diff_1_3  # [4]
    loss = torch.clamp(margin, min=0.01).mean()
    return loss


if __name__ == "__main__":
    num_nodes = 5000
    input_dims = [512, 128, 256]
    hidden_dims = [256, 512, 1024]

    output_dim = 200
    gnn_type = 'gcn'
    alpha = 0.6

    x1 = torch.randn((num_nodes, input_dims[0]))
    x2 = torch.randn((num_nodes, input_dims[1]))
    x3 = torch.randn((num_nodes, input_dims[2]))
    features = [x1, x2, x3]

    adj_matrix = torch.randint(0, 2, (num_nodes, num_nodes))
    src, dst = adj_matrix.nonzero(as_tuple=True)
    graph = dgl.graph((src, dst))

    model = mGCN(input_dims=input_dims, hidden_dims=hidden_dims, output_dim=output_dim, gnn_type=gnn_type, alpha=alpha)

    output = model(features, graph)
    print(output.shape)
