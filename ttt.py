import numpy as np
from torch_geometric.graphgym.register import loss_dict
from torch_geometric.utils import dropout_adj
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
#测试时训练主函数

#TODO
# from cluster.kmeans import kmeans

#结果
# 训练过程 w cluster loss
# TTT cluster loss
#
# 训练过程 w/o cluster loss
# TTT cluster loss
#
# 训练过程 w cluster loss




def soft_kmeans(features: torch.Tensor,  init_centers: torch.Tensor,  alpha: float = 10.0,  max_iter: int = 5,  detach_centers: bool = True):
    centers = init_centers
    N, d = features.size()
    K = centers.size(0)

    for _ in range(max_iter):
        # 根据当前 centers 计算每个节点到各簇中心的距离
        dist = torch.cdist(features, centers, p=2)
        # softmax
        cluster_probs = F.softmax(-alpha * dist, dim=1)

        # 更新聚类中心，根据软分配做加权平均
        numerator = torch.einsum('nk,nd->kd', cluster_probs, features)
        denom = cluster_probs.sum(dim=0, keepdim=True)
        new_centers = numerator / (denom.transpose(0,1) + 1e-9)

        if detach_centers:
            centers = new_centers.detach()
        else:
            centers = new_centers

    return cluster_probs, centers


def compute_soft_kmeans_align_loss(node_patterns, expert_weights, num_clusters=5, alpha=10.0, max_iter=5, use_match_matrix=False, device='cuda', detach_centers=True, plot = False):
    N = node_patterns.size(0)

    # 随机初始化两套聚类中心
    centers_np = torch.randn(num_clusters, node_patterns.size(1), device=device, requires_grad=(not detach_centers))
    centers_ew = torch.randn(num_clusters, expert_weights.size(1), device=device, requires_grad=(not detach_centers))

    # 聚类
    P, centers_np = soft_kmeans(node_patterns, centers_np, alpha=alpha, max_iter=max_iter, detach_centers=detach_centers)
    # 聚类
    Q, centers_ew = soft_kmeans(expert_weights, centers_ew, alpha=alpha, max_iter=max_iter, detach_centers=detach_centers)
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
        kl_per_node = torch.sum(P * torch.log((P + 1e-9) / (Q_hat + 1e-9)), dim=1)
        kl_loss = kl_per_node.mean()
        return kl_loss, centers_np, centers_ew, M_prob
    else:
        # 无匹配矩阵的对齐
        kl_per_node = torch.sum(P * torch.log((P + 1e-9) / (Q + 1e-9)), dim=1)
        kl_loss = kl_per_node.mean()
        return kl_loss, centers_np, centers_ew, None

