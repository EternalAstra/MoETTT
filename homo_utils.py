import torch
from torch_geometric.utils import  remove_self_loops
from torch_scatter import scatter_add

def compute_attributes_feature_homo(data, similarity_threshold=0.8):
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_value = torch.ones([edge_index.size(1)], device=edge_index.device)
    num_labels = torch.max(data.y).item() + 1
    num_nodes = data.y.shape[0]
    row, col = edge_index[0], edge_index[1]
    # 每个节点的度数 deg，即与每个节点相连的边的数量
    deg = scatter_add(edge_value, row, dim=0, dim_size=num_nodes)
    # 计算连接的节点对在特征空间中的相似度
    # 这里我们使用余弦相似度，也可以选择其他相似度度量
    # 首先对特征进行归一化
    normalized_features = data.x / data.x.norm(dim=1, keepdim=True)
    # 然后计算连接的节点对的归一化特征的点积
    edge_feature_similarity = (normalized_features[row] * normalized_features[col]).sum(dim=1)
    # 你可以设定一个阈值来决定什么程度的相似度被认为是同质的
    # 例如，相似度大于0.9的节点对被认为是同质的
    edge_homo_value = (edge_feature_similarity > similarity_threshold).int()
    # 计算每个节点的同质性比例
    homo_ratio = scatter_add(edge_homo_value, row, dim=0, dim_size=num_nodes)
    homo_ratio = torch.squeeze(homo_ratio)
    # 计算每个节点的特征同质性值
    results = homo_ratio / deg
    return results

def compute_embeddings_feature_homo(data, similarity_threshold=0.8):
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_value = torch.ones([edge_index.size(1)], device=edge_index.device)
    num_labels = torch.max(data.y).item() + 1
    num_nodes = data.y.shape[0]
    row, col = edge_index[0], edge_index[1]
    # 每个节点的度数 deg，即与每个节点相连的边的数量
    deg = scatter_add(edge_value, row, dim=0, dim_size=num_nodes)

    # 加载node embeddings
    embeddings = torch.load('embeddings/Chameleon_embeddings.pt').to(device=edge_index.device)
    normalized_features = embeddings / embeddings.norm(dim=1, keepdim=True)

    # 然后计算连接的节点对的归一化特征的点积
    edge_feature_similarity = (normalized_features[row] * normalized_features[col]).sum(dim=1)
    # 你可以设定一个阈值来决定什么程度的相似度被认为是同质的
    edge_homo_value = (edge_feature_similarity > similarity_threshold).int()
    # 计算每个节点的同质性比例
    homo_ratio = scatter_add(edge_homo_value, row, dim=0, dim_size=num_nodes)
    homo_ratio = torch.squeeze(homo_ratio)
    # 计算每个节点的特征同质性值
    results = homo_ratio / deg
    return results

def get_homophily_split(data,split):
    edge_index = data.graph['edge_index']
    edge_index, _ = remove_self_loops(edge_index)
    edge_value = torch.ones([edge_index.size(1)], device=edge_index.device)
    num_labels = torch.max(data.label).item() + 1
    num_nodes = data.label.shape[0]
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_value, row, dim=0, dim_size=num_nodes)  # 每个节点的度数 deg，即与每个节点相连的边的数量
    edge_homo_value = (data.label[row] == data.label[col]).int()
    homo_ratio = scatter_add(edge_homo_value, row, dim=0, dim_size=num_nodes)
    homo_ratio = torch.squeeze(homo_ratio)
    results = homo_ratio / deg
    train_result = results.tolist()

    split = split.tolist()
    split1 = []
    split2 = []

    for i in range(0,len(split)):
        if train_result[split[i]] < 0.5:
            split1.append(split[i])
        else:
            split2.append(split[i])

    return torch.tensor(split1),torch.tensor(split2)

def get_feature_homophily_split(data,split):
    similarity_threshold = 0.5
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_value = torch.ones([edge_index.size(1)], device=edge_index.device)
    num_labels = torch.max(data.y).item() + 1
    num_nodes = data.y.shape[0]
    row, col = edge_index[0], edge_index[1]
    # 每个节点的度数 deg，即与每个节点相连的边的数量
    deg = scatter_add(edge_value, row, dim=0, dim_size=num_nodes)
    # 计算连接的节点对在特征空间中的相似度
    # 这里我们使用余弦相似度，也可以选择其他相似度度量
    # 首先对特征进行归一化
    normalized_features = data.x / data.x.norm(dim=1, keepdim=True)
    # 然后计算连接的节点对的归一化特征的点积
    edge_feature_similarity = (normalized_features[row] * normalized_features[col]).sum(dim=1)
    # 你可以设定一个阈值来决定什么程度的相似度被认为是同质的
    # 例如，相似度大于0.9的节点对被认为是同质的
    edge_homo_value = (edge_feature_similarity > similarity_threshold).int()
    # 计算每个节点的同质性比例
    homo_ratio = scatter_add(edge_homo_value, row, dim=0, dim_size=num_nodes)
    homo_ratio = torch.squeeze(homo_ratio)
    results = homo_ratio / deg
    train_result = results.tolist()

    split = split.tolist()
    split1 = []
    split2 = []
    split3 = []

    for i in range(0,len(split)):
        if train_result[split[i]] < 0.33:
            split1.append(split[i])
        elif train_result[split[i]] < 0.66:
            split2.append(split[i])
        else:
            split3.append(split[i])

    return torch.tensor(split1),torch.tensor(split2),torch.tensor(split3)



import torch
from torch_scatter import scatter_add

def get_feature_homophily_split(data, test_indices):
    edge_index = data.graph['edge_index']
    edge_index, _ = remove_self_loops(edge_index)
    edge_value = torch.ones([edge_index.size(1)], device=edge_index.device)
    num_labels = torch.max(data.label).item() + 1
    num_nodes = data.label.shape[0]
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_value, row, dim=0, dim_size=num_nodes)  # 每个节点的度数 deg

    # 计算特征余弦相似度
    feature_similarity = torch.nn.functional.cosine_similarity(data.graph['node_feat'][row], data.graph['node_feat'][col], dim=1)
    avg_feature_similarity = scatter_add(feature_similarity, row, dim=0, dim_size=num_nodes) / deg
    avg_feature_similarity = torch.squeeze(avg_feature_similarity)

    # 仅对输入列表中的节点进行操作
    test_avg_feature_similarity = avg_feature_similarity[test_indices]
    sorted_indices = torch.argsort(test_avg_feature_similarity, descending=False)  # 按照特征相似度值从低到高排序
    sorted_test_indices = test_indices[sorted_indices]

    # 将排序后的节点等分为三组
    total_nodes = sorted_test_indices.shape[0]
    group_size = total_nodes // 5
    test_idx_1 = sorted_test_indices[:group_size]
    test_idx_2 = sorted_test_indices[group_size:2 * group_size]
    test_idx_3 = sorted_test_indices[2 * group_size:3 * group_size]
    test_idx_4 = sorted_test_indices[3 * group_size:4 * group_size]
    test_idx_5 = sorted_test_indices[4 * group_size:]

    # 计算每个分组的特征相似度区间
    test_avg_feature_similarity_sorted = avg_feature_similarity[sorted_indices]
    min_feature_similarity_1, max_feature_similarity_1 = test_avg_feature_similarity_sorted[:group_size].min().item(), test_avg_feature_similarity_sorted[:group_size].max().item()
    min_feature_similarity_2, max_feature_similarity_2 = test_avg_feature_similarity_sorted[group_size:2 * group_size].min().item(), test_avg_feature_similarity_sorted[group_size:2 * group_size].max().item()
    min_feature_similarity_3, max_feature_similarity_3 = test_avg_feature_similarity_sorted[2 * group_size:3 * group_size].min().item(), test_avg_feature_similarity_sorted[2 * group_size:3 * group_size].max().item()
    min_feature_similarity_4, max_feature_similarity_4 = test_avg_feature_similarity_sorted[3 * group_size:4 * group_size].min().item(), test_avg_feature_similarity_sorted[3 * group_size:4 * group_size].max().item()
    min_feature_similarity_5, max_feature_similarity_5 = test_avg_feature_similarity_sorted[4 * group_size:].min().item(), test_avg_feature_similarity_sorted[4 * group_size:].max().item()

    # 返回分组及其特征相似度区间
    return (test_idx_1, (min_feature_similarity_1, max_feature_similarity_1)), (test_idx_2, (min_feature_similarity_2, max_feature_similarity_2)), (test_idx_3, (min_feature_similarity_3, max_feature_similarity_3)), (test_idx_4, (min_feature_similarity_4, max_feature_similarity_4)), (test_idx_5, (min_feature_similarity_5, max_feature_similarity_5))





def get_group_split(data, test_indices):
    edge_index = data.graph['edge_index']
    edge_index, _ = remove_self_loops(edge_index)
    edge_value = torch.ones([edge_index.size(1)], device=edge_index.device)
    num_labels = torch.max(data.label).item() + 1
    num_nodes = data.label.shape[0]
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_value, row, dim=0, dim_size=num_nodes)  # 每个节点的度数 deg
    edge_homo_value = (data.label[row] == data.label[col]).int()
    homo_ratio = scatter_add(edge_homo_value, row, dim=0, dim_size=num_nodes)
    homo_ratio = torch.squeeze(homo_ratio)
    results = homo_ratio / deg
    # 仅对输入列表中的节点进行操作
    test_results = results[test_indices]
    sorted_indices = torch.argsort(test_results, descending=False)  # 按照homophily值从高到低排序
    sorted_test_indices = test_indices[sorted_indices]

    # 将排序后的节点等分为三组
    total_nodes = sorted_test_indices.shape[0]
    group_size = total_nodes // 5
    test_idx_1 = sorted_test_indices[:group_size]
    test_idx_2 = sorted_test_indices[group_size:2 * group_size]
    test_idx_3 = sorted_test_indices[2 * group_size:3 * group_size]
    test_idx_4 = sorted_test_indices[3 * group_size:4 * group_size]
    test_idx_5 = sorted_test_indices[4 * group_size:]

    # 计算每个分组的homophily区间
    test_results_sorted = test_results[sorted_indices]
    min_homophily_1, max_homophily_1 = test_results_sorted[:group_size].min().item(), test_results_sorted[:group_size].max().item()
    min_homophily_2, max_homophily_2 = test_results_sorted[group_size:2 * group_size].min().item(), test_results_sorted[group_size:2 * group_size].max().item()
    min_homophily_3, max_homophily_3 = test_results_sorted[2 * group_size:3 * group_size].min().item(), test_results_sorted[2 * group_size:3 * group_size].max().item()
    min_homophily_4, max_homophily_4 = test_results_sorted[3 * group_size:4 * group_size].min().item(), test_results_sorted[3 * group_size:4 * group_size].max().item()
    min_homophily_5, max_homophily_5 = test_results_sorted[4 * group_size:].min().item(), test_results_sorted[4 * group_size:].max().item()

    # 返回分组及其homophily区间
    return (test_idx_1, (min_homophily_1, max_homophily_1)), (test_idx_2, (min_homophily_2, max_homophily_2)), (test_idx_3, (min_homophily_3, max_homophily_3)), (test_idx_4, (min_homophily_4, max_homophily_4)), (test_idx_5, (min_homophily_5, max_homophily_5))


def get_feature_group_split(data, test_indices):
    similarity_threshold = 0.4
    edge_index = data.graph['edge_index']
    edge_value = torch.ones([edge_index.size(1)], device=edge_index.device)
    num_labels = torch.max(data.label).item() + 1
    num_nodes = data.label.shape[0]
    row, col = edge_index[0], edge_index[1]
    # 每个节点的度数 deg，即与每个节点相连的边的数量
    deg = scatter_add(edge_value, row, dim=0, dim_size=num_nodes)
    # 计算连接的节点对在特征空间中的相似度
    # 首先对特征进行归一化
    normalized_features = data.graph['node_feat'] / data.graph['node_feat'].norm(dim=1, keepdim=True)
    # 然后计算连接的节点对的归一化特征的点积
    edge_feature_similarity = (normalized_features[row] * normalized_features[col]).sum(dim=1)
    # 你可以设定一个阈值来决定什么程度的相似度被认为是同质的
    # 例如，相似度大于0.9的节点对被认为是同质的
    edge_homo_value = (edge_feature_similarity > similarity_threshold).int()
    # 计算每个节点的同质性比例
    homo_ratio = scatter_add(edge_homo_value, row, dim=0, dim_size=num_nodes)
    homo_ratio = torch.squeeze(homo_ratio)
    results = homo_ratio / deg
    # 仅对输入列表中的节点进行操作
    test_results = results[test_indices]
    sorted_indices = torch.argsort(test_results, descending=False)  # 按照homophily值从高到低排序
    sorted_test_indices = test_indices[sorted_indices]

    # 将排序后的节点等分为三组
    total_nodes = sorted_test_indices.shape[0]
    group_size = total_nodes // 5
    test_idx_1 = sorted_test_indices[:group_size]
    test_idx_2 = sorted_test_indices[group_size:2 * group_size]
    test_idx_3 = sorted_test_indices[2 * group_size:3 * group_size]
    test_idx_4 = sorted_test_indices[3 * group_size:4 * group_size]
    test_idx_5 = sorted_test_indices[4 * group_size:]

    # 计算每个分组的homophily区间
    test_results_sorted = test_results[sorted_indices]
    min_homophily_1, max_homophily_1 = test_results_sorted[:group_size].min().item(), test_results_sorted[:group_size].max().item()
    min_homophily_2, max_homophily_2 = test_results_sorted[group_size:2 * group_size].min().item(), test_results_sorted[group_size:2 * group_size].max().item()
    min_homophily_3, max_homophily_3 = test_results_sorted[2 * group_size:3 * group_size].min().item(), test_results_sorted[2 * group_size:3 * group_size].max().item()
    min_homophily_4, max_homophily_4 = test_results_sorted[3 * group_size:4 * group_size].min().item(), test_results_sorted[3 * group_size:4 * group_size].max().item()
    min_homophily_5, max_homophily_5 = test_results_sorted[4 * group_size:].min().item(), test_results_sorted[4 * group_size:].max().item()

    # 返回分组及其homophily区间
    return (test_idx_1, (min_homophily_1, max_homophily_1)), (test_idx_2, (min_homophily_2, max_homophily_2)), (test_idx_3, (min_homophily_3, max_homophily_3)), (test_idx_4, (min_homophily_4, max_homophily_4)), (test_idx_5, (min_homophily_5, max_homophily_5))



def get_deg_group_split(data, test_indices):
    edge_index = data.graph['edge_index']
    edge_index, _ = remove_self_loops(edge_index)
    edge_value = torch.ones([edge_index.size(1)], device=edge_index.device)
    num_labels = torch.max(data.label).item() + 1
    num_nodes = data.label.shape[0]
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_value, row, dim=0, dim_size=num_nodes)  # 每个节点的度数 deg
    results = deg
    # 仅对输入列表中的节点进行操作
    test_results = results[test_indices]
    sorted_indices = torch.argsort(test_results, descending=False)  # 按照homophily值从高到低排序
    sorted_test_indices = test_indices[sorted_indices]

    # 将排序后的节点等分为三组
    total_nodes = sorted_test_indices.shape[0]
    group_size = total_nodes // 5
    test_idx_1 = sorted_test_indices[:group_size]
    test_idx_2 = sorted_test_indices[group_size:2 * group_size]
    test_idx_3 = sorted_test_indices[2 * group_size:3 * group_size]
    test_idx_4 = sorted_test_indices[3 * group_size:4 * group_size]
    test_idx_5 = sorted_test_indices[4 * group_size:]

    # 计算每个分组的homophily区间
    test_results_sorted = test_results[sorted_indices]
    min_deg_1, max_deg_1 = test_results_sorted[:group_size].min().item(), test_results_sorted[:group_size].max().item()
    min_deg_2, max_deg_2 = test_results_sorted[group_size:2 * group_size].min().item(), test_results_sorted[group_size:2 * group_size].max().item()
    min_deg_3, max_deg_3 = test_results_sorted[2 * group_size:3 * group_size].min().item(), test_results_sorted[2 * group_size:3 * group_size].max().item()
    min_deg_4, max_deg_4 = test_results_sorted[3 * group_size:4 * group_size].min().item(), test_results_sorted[3 * group_size:4 * group_size].max().item()
    min_deg_5, max_deg_5 = test_results_sorted[4 * group_size:].min().item(), test_results_sorted[4 * group_size:].max().item()

    # 返回分组及其homophily区间
    return (test_idx_1, (min_deg_1, max_deg_1)), (test_idx_2, (min_deg_2, max_deg_2)), (test_idx_3, (min_deg_3, max_deg_3)), (test_idx_4,(min_deg_4, max_deg_4)), (test_idx_5, (min_deg_5, max_deg_5))





def compute_homo(edge_index,labels):
    edge_index, _ = remove_self_loops(edge_index)
    edge_value = torch.ones([edge_index.size(1)], device=edge_index.device)
    num_nodes = labels.shape[0]
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_value, row, dim=0, dim_size=num_nodes)  # 每个节点的度数 deg
    edge_homo_value = (labels[row] == labels[col]).int()
    homo_ratio = scatter_add(edge_homo_value, row, dim=0, dim_size=num_nodes)
    homo_ratio = torch.squeeze(homo_ratio)
    results = homo_ratio / deg
    return results


