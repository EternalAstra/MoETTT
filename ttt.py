import numpy as np
from torch_geometric.graphgym.register import loss_dict
from torch_geometric.utils import dropout_adj
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans



def compute_soft_kmeans_align_loss(node_patterns, expert_weights, num_clusters=10, alpha=10.0, max_iter=20, use_match_matrix=True, device='cuda',  plot = False, idx=None):
    if idx is not None:
        node_patterns = node_patterns[idx]
        expert_weights = expert_weights[idx]


    # Initialize KMeans for both sets of clusters (on CPU for fitting)
    kmeans_np = KMeans(n_clusters=num_clusters, max_iter=max_iter)
    kmeans_ew = KMeans(n_clusters=num_clusters, max_iter=max_iter)

    # Fit KMeans to the node patterns and expert weights
    kmeans_np.fit(node_patterns.cpu().detach().numpy())
    kmeans_ew.fit(expert_weights.cpu().detach().numpy())

    # Get the cluster centers from KMeans
    centers_np = torch.tensor(kmeans_np.cluster_centers_, device=device)
    centers_ew = torch.tensor(kmeans_ew.cluster_centers_, device=device)

    # Calculate the cluster probabilities using softmax on distances without numpy
    # Using broadcasting to calculate distances
    dist_np = torch.cdist(node_patterns.unsqueeze(0), centers_np.unsqueeze(0)).squeeze(0)
    dist_ew = torch.cdist(expert_weights.unsqueeze(0), centers_ew.unsqueeze(0)).squeeze(0)

    # Calculate softmax probabilities
    P = torch.softmax(-alpha * dist_np, dim=1)
    Q = torch.softmax(-alpha * dist_ew, dim=1)

    if  plot:
        # 降维并可视化
        tsne = TSNE(n_components=2, random_state=33)
        node_patterns_2d = tsne.fit_transform(node_patterns.detach().cpu().numpy())
        expert_weights_2d = tsne.fit_transform(expert_weights.detach().cpu().numpy())

        plt.figure(figsize=(12, 6))

        # 为每个簇分配一个颜色
        colors = plt.cm.get_cmap('tab10', num_clusters)  # 使用colormap
        cluster_labels = torch.argmax(P, dim=1).cpu().numpy()  # 获取簇标签

        # 可视化节点模式
        plt.subplot(1, 2, 1)
        for i in range(num_clusters):
            plt.scatter(node_patterns_2d[cluster_labels == i, 0],
                        node_patterns_2d[cluster_labels == i, 1],
                        color=colors(i), alpha=0.5, label=f'Cluster {i}')
        plt.title('Node Patterns t-SNE')
        plt.xlabel('t-SNE component 1')
        plt.ylabel('t-SNE component 2')
        plt.legend()

        # 可视化专家权重
        plt.subplot(1, 2, 2)
        cluster_labels_ew = torch.argmax(Q, dim=1).cpu().numpy()  # 获取簇标签
        for i in range(num_clusters):
            plt.scatter(expert_weights_2d[cluster_labels_ew == i, 0],
                        expert_weights_2d[cluster_labels_ew == i, 1],
                        color=colors(i), alpha=0.5, label=f'Expert Cluster {i}')
        plt.title('Expert Weights t-SNE')
        plt.xlabel('t-SNE component 1')
        plt.ylabel('t-SNE component 2')
        plt.legend()

        plt.tight_layout()
        plt.show()

    if use_match_matrix:
        # 计算成本矩阵
        cost_matrix = torch.zeros(num_clusters, num_clusters, device=device)

        for i in range(num_clusters):
            for j in range(num_clusters):
                # 计算平均KL散度
                cost_matrix[i, j] = torch.mean(P[:, i] * torch.log((P[:, i] + 1e-9) / (Q[:, j] + 1e-9)))

        # 使用匈牙利算法进行最佳匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())

        # 创建匹配后的分布矩阵
        M_prob = torch.zeros(num_clusters, num_clusters, device=device)
        for i, j in zip(row_ind, col_ind):
            M_prob[i, j] = 1.0  # 赋值给最佳匹配的位置

        # 计算Q_hat
        Q_hat = Q @ M_prob.T

        # 计算KL散度
        kl_loss = F.kl_div(Q_hat.log() + 1e-9, P, reduction='batchmean')
        return kl_loss
    else:
        kl_loss = F.kl_div(Q.log() + 1e-9, P, reduction='batchmean')
        return kl_loss

