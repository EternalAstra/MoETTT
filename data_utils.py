import os
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
import numpy as np
import scipy.sparse as sp
import networkx as nx
import sys
import pickle as pkl
from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric.utils import  remove_self_loops
from torch_scatter import scatter_add
from models import GatingNetwork
from collections import defaultdict
device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

splits_drive_url = {
    'snap-patents' : '12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N',
    'pokec' : '1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_',
}

dataset_drive_url = {
    'twitch-gamer_feat' : '1fA9VIIEI8N0L27MSQfcBzJgRQLvSbrvR',
    'twitch-gamer_edges' : '1XLETC6dG3lVl7kDmytEJ52hvDMVdxnZ0',
    'snap-patents' : '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia',
    'pokec' : '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y',
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ',
    'wiki_views': '1p5DlVHrnFgYm3VsNIzahSsvCD424AyvP', # Wiki 1.9M
    'wiki_edges': '14X7FlkjrlUgmnsYtPwdh-gGuFla4yb5u', # Wiki 1.9M
    'wiki_features': '1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK' # Wiki 1.9M
}


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_fixed_splits(dataset, split):
    """ loads saved fixed splits for dataset
    """
    name = dataset
    splits_file_path = './data/splits/' + name + '_split_0.6_0.2_' + str(split) + '.npz'

    with np.load(splits_file_path) as splits_file:
        train_mask = splits_file['train_mask']
        val_mask = splits_file['val_mask']
        test_mask = splits_file['test_mask']

    idx_train = np.where(train_mask == 1)[0]
    idx_val = np.where(val_mask == 1)[0]
    idx_test = np.where(test_mask == 1)[0]

    split_idx = {'train': idx_train,
                 'valid': idx_val,
                 'test': idx_test}

    return split_idx


def get_homophily_shift(data):
    # 提取边索引并移除自环
    edge_index = data.graph['edge_index']
    edge_index, _ = remove_self_loops(edge_index)
    edge_value = torch.ones([edge_index.size(1)], device=edge_index.device)
    num_nodes = data.label.shape[0]

    # 计算每个节点的度数
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_value, row, dim=0, dim_size=num_nodes)  # 各节点的度数

    # 计算每个节点的homophily值（同类邻居比例）
    edge_homo_value = (data.label[row] == data.label[col]).int()  # 边是否连接同类节点（1表示是）
    homo_ratio = scatter_add(edge_homo_value, row, dim=0, dim_size=num_nodes)  # 各节点的同类边数
    homo_ratio = torch.squeeze(homo_ratio)
    homo_ratio = homo_ratio / deg  # 归一化为比例（处理度数为0的情况避免除零错误）

    # 步骤1：根据homophily阈值划分训练候选集和测试集
    # 训练候选集：homophily > 0.5的节点
    train_candidate_mask = homo_ratio > 0.5
    train_candidates = torch.where(train_candidate_mask)[0]  # 训练候选集索引

    # 测试集：homophily <= 0.5的节点
    test_mask = ~train_candidate_mask
    test_idx = torch.where(test_mask)[0]  # 测试集索引


    # 步骤2：从训练候选集中划分训练集和验证集（按split中的比例）
    # 假设split格式为 {'train': 0.8, 'valid': 0.2}，表示训练候选集的80%作为训练集，20%作为验证集
    train_prop = 0.8
    valid_prop = 0.2

    # 打乱训练候选集顺序以保证随机性
    shuffled_train_candidates = train_candidates[torch.randperm(len(train_candidates))]
    num_train_candidates = len(shuffled_train_candidates)

    # 计算训练集和验证集的数量
    train_num = int(num_train_candidates * train_prop)
    valid_num = num_train_candidates - train_num  # 剩余作为验证集（兼容split中valid未显式指定的情况）

    # 划分训练集和验证集
    train_idx = shuffled_train_candidates[:train_num]
    valid_idx = shuffled_train_candidates[train_num:train_num + valid_num]

    # 返回划分结果（确保索引为长整型张量）
    split_idx = {
        'train': train_idx.long(),
        'valid': valid_idx.long(),
        'test': test_idx.long()
    }
    return split_idx

# 根据homophily值划分数据集
def get_homophily_split(data, split):

    edge_index = data.graph['edge_index']
    edge_index, _ = remove_self_loops(edge_index)
    edge_value = torch.ones([edge_index.size(1)], device=edge_index.device)
    num_nodes = data.label.shape[0]
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_value, row, dim=0, dim_size=num_nodes)  # 每个节点的度数 deg
    edge_homo_value = (data.label[row] == data.label[col]).int()
    homo_ratio = scatter_add(edge_homo_value, row, dim=0, dim_size=num_nodes)
    homo_ratio = torch.squeeze(homo_ratio)
    homo_ratio = homo_ratio / deg

    # 排序：根据homophily值降序排列
    sorted_indices = torch.argsort(homo_ratio, descending=True)  # 从高到低排序节点索引

    #如果 homophily大于 0.5 进入训练集，下于0.5进入测试集，把训练集中的20%作为valid

    # 按比例划分为训练集、验证集和测试集
    train_prop, valid_prop = split['train'], split['valid']
    train_num = int(num_nodes * train_prop)
    valid_num = int(num_nodes * valid_prop)
    test_num = num_nodes - train_num - valid_num

    train_idx = sorted_indices[:train_num]
    valid_idx = sorted_indices[train_num:train_num + valid_num]
    test_idx = sorted_indices[train_num + valid_num:]

    # 返回划分的索引
    split_idx = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
    return split_idx


def rand_train_test_idx(label, train_prop=.48, valid_prop=.32, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx


def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on

    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label


def to_sparse_tensor(edge_index, edge_feat, num_nodes):
    """ converts the edge_index into SparseTensor
    """
    num_edges = edge_index.size(1)

    (row, col), N, E = edge_index, num_nodes, num_edges
    perm = (col * N + row).argsort()
    row, col = row[perm], col[perm]

    value = edge_feat[perm]
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(N, N), is_sorted=True)

    # Pre-process some important attributes.
    adj_t.storage.rowptr()
    adj_t.storage.csr2csc()

    return adj_t

def normalize(edge_index):
    """ normalizes the edge_index
    """
    adj_t = edge_index.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    return adj_t

def data_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx



def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct)) / len(correct))

    return sum(acc_list) / len(acc_list)

def eval_ST(y_true, y_1,y_2):
    acc_list =[]
    y_true = y_true.detach().cpu().numpy()
    y_1 = y_1.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    y_2 = y_2.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_1.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        same = y_1[is_labeled, i] == y_2[is_labeled, i]


    for i in range(y_1.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = np.logical_and(y_true[is_labeled, i] == y_1[is_labeled, i], y_true[is_labeled, i] == y_2[is_labeled, i])
        acc_list.append(float(np.sum(correct)) / float(np.sum(same)))
    return sum(acc_list) / len(acc_list)

def eval_rocauc(y_true, y_pred):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = F.softmax(y_pred, dim=-1).detach().cpu().numpy()
    num_classes = y_pred.shape[1]

    if y_true.shape[1] == 1 and num_classes > 2:
        # multi-class with class-index labels: use OvR AUC
        y_true_flat = y_true[:, 0].astype(int)
        try:
            score = roc_auc_score(y_true_flat, y_pred, multi_class='ovr', average='macro',
                                  labels=list(range(num_classes)))
            rocauc_list.append(score)
        except ValueError:
            return 0.0
    elif y_true.shape[1] == 1:
        # binary classification
        y_pred_bin = y_pred[:, 1]
        is_labeled = y_true[:, 0] == y_true[:, 0]
        if np.sum(y_true[:, 0] == 1) > 0 and np.sum(y_true[:, 0] == 0) > 0:
            score = roc_auc_score(y_true[is_labeled, 0], y_pred_bin[is_labeled])
            rocauc_list.append(score)
    else:
        # multi-label: one column per class
        for i in range(y_true.shape[1]):
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
                is_labeled = y_true[:, i] == y_true[:, i]
                score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
                rocauc_list.append(score)

    if len(rocauc_list) == 0:
        return 0.0

    return sum(rocauc_list) / len(rocauc_list)


def eval_f1(y_true, y_pred):
    """Compute macro F1 score."""
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    f1_list = []
    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        f1 = f1_score(y_true[is_labeled, i], y_pred[is_labeled, i], average='macro')
        f1_list.append(f1)

    return sum(f1_list) / len(f1_list)



@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func):

    model.eval()
    out = model(dataset)

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])

    return train_acc, valid_acc, test_acc, out

@torch.no_grad()
def group_evaluate(model, dataset, test_idx_1, test_idx_2, test_idx_3,test_idx_4, test_idx_5, eval_func):

    model.eval()
    out = model(dataset)

    test1_acc = eval_func(
        dataset.label[test_idx_1], out[test_idx_1])
    test2_acc = eval_func(
        dataset.label[test_idx_2], out[test_idx_2])
    test3_acc = eval_func(
        dataset.label[test_idx_3], out[test_idx_3])
    test4_acc = eval_func(
        dataset.label[test_idx_4], out[test_idx_4])
    test5_acc = eval_func(
        dataset.label[test_idx_5], out[test_idx_5])
    return test1_acc, test2_acc, test3_acc,test4_acc, test5_acc


### label homophily choice ###
def label_homophily_adapt_predict_merge(data,out1,out2):
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
    results = results.tolist()
    out1 = out1.tolist()
    out2 = out2.tolist()
    out = []
    for i in range(0,len(results)):
        if results[i] > 0.5:
            out.append(out1[i])
        else:
            out.append(out2[i])

    return  torch.tensor(out)


@torch.no_grad()
def adapt_predict_merge(data, out1, out2, similarity_threshold=0.5):
    edge_index = data.graph['edge_index']
    edge_index, _ = remove_self_loops(edge_index)
    num_nodes = data.label.shape[0]
    row, col = edge_index[0], edge_index[1]

    # 将out1和out2转换为单位向量，以便计算余弦相似度
    out1_norm = out1 / out1.norm(dim=1, keepdim=True)
    out2_norm = out2 / out2.norm(dim=1, keepdim=True)

    # 计算所有边的相似度
    similarity = (out1_norm[row] * out2_norm[col]).sum(dim=1)

    # 判定每条边是同质边还是异质边
    homo_edges = similarity > similarity_threshold

    # 计算每个节点的同质边数量
    homo_edge_count = scatter_add(homo_edges.float(), row, dim=0, dim_size=num_nodes)

    # 计算每个节点的度
    deg = scatter_add(torch.ones(edge_index.size(1), device=edge_index.device), row, dim=0, dim_size=num_nodes)

    # 计算每个节点的metric
    metric =  homo_edge_count /  deg

    # 根据metric选择输出结果
    out = torch.zeros_like(out1)
    for i in range(num_nodes):
        if metric[i] > 0.5:
            out[i] = out1[i]
        else:
            out[i] = out2[i]
    return out


@torch.no_grad()
def HC_evaluate(model1,model2, dataset, split_idx, eval_func,similarity_threshold =0.5):

    model1.eval()
    model2.eval()
    out1 = model1(dataset)
    out2 = model2(dataset)
    out = label_homophily_adapt_predict_merge(dataset,out1,out2)

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])

    return train_acc, valid_acc, test_acc, out


@torch.no_grad()
def ST_evaluate(model1,model2, dataset, split_idx, eval_func,similarity_threshold =0.5):

    model1.eval()
    model2.eval()
    out1 = model1(dataset)
    out2 = model2(dataset)


    train_acc = eval_func(
        dataset.label[split_idx['train']],out1[split_idx['train']], out2[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']],out1[split_idx['valid']], out2[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']],out1[split_idx['test']], out2[split_idx['test']])

    return train_acc, valid_acc, test_acc,out1



@torch.no_grad()
def mean_moe_evaluate(model1,model2,model3,model4,model5,model6,model7, dataset, split_idx, eval_func):

    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    model6.eval()
    model7.eval()
    out1 = model1(dataset)
    out2 = model2(dataset)
    out3 = model3(dataset)
    out4 = model4(dataset)
    out5 = model5(dataset)
    out6 = model6(dataset)
    out7 = model7(dataset)


    out = mean_moe_predict_merge(out1,out2,out3,out4,out5,out6,out7)

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])

    return train_acc, valid_acc, test_acc, out


def mean_moe_predict_merge(out1,out2,out3,out4,out5,out6,out7):
    out = (out1 + out2 + out3 + out4 + out5 + out6 + out7) / 7
    return  torch.tensor(out)












def gen_normalized_adjs(dataset):
    """ returns the normalized adjacency matrix
    """
    row, col = dataset.graph['edge_index']
    N = dataset.graph['num_nodes']
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    deg = adj.sum(dim=1).to(torch.float)
    D_isqrt = deg.pow(-0.5)
    D_isqrt[D_isqrt == float('inf')] = 0

    DAD = D_isqrt.view(-1,1) * adj * D_isqrt.view(1,-1)
    DA = D_isqrt.view(-1,1) * D_isqrt.view(-1,1) * adj
    AD = adj * D_isqrt.view(1,-1) * D_isqrt.view(1,-1)
    return DAD, DA, AD

def load_data_new(dataset_str, split = 0):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    # print('dataset_str', dataset_str)
    # print('split', split)
    if dataset_str in ['citeseer', 'cora', 'pubmed']:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file(
            "data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(
                min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        splits_file_path = 'splits/' + dataset_str + \
            '_split_0.6_0.2_' + str(split) + '.npz'

        with np.load(splits_file_path) as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']

        idx_train = list(np.where(train_mask == 1)[0])
        idx_val = list(np.where(val_mask == 1)[0])
        idx_test = list(np.where(test_mask == 1)[0])

        no_label_nodes = []
        if dataset_str == 'citeseer':  # citeseer has some data with no label
            for i in range(len(labels)):
                if sum(labels[i]) < 1:
                    labels[i][0] = 1
                    no_label_nodes.append(i)

            for n in no_label_nodes:  # remove unlabel nodes from train/val/test
                if n in idx_train:
                    idx_train.remove(n)
                if n in idx_val:
                    idx_val.remove(n)
                if n in idx_test:
                    idx_test.remove(n)

    elif dataset_str in ['chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin']:
        graph_adjacency_list_file_path = os.path.join(
            'new_data', dataset_str, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join('new_data', dataset_str,
                                                                f'out1_node_feature_label.txt')

        # graph_dict = defaultdict(list)
        # with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        #     graph_adjacency_list_file.readline()
        #     for line in graph_adjacency_list_file:
        #         line = line.rstrip().split('\t')
        #         assert (len(line) == 2)
        #         graph_dict[int(line[0])].append(int(line[1]))
        #         graph_dict[int(line[1])].append(int(line[0]))
        #
        # # print(sorted(graph_dict))
        # graph_dict_ordered = defaultdict(list)
        # for key in sorted(graph_dict):
        #     graph_dict_ordered[key] = graph_dict[key]
        #     graph_dict_ordered[key].sort()
        #
        # adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph_dict_ordered))
        # 存储边关系的列表
        edges = []

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            # 跳过第一行（header）
            graph_adjacency_list_file.readline()

            # 逐行读取边数据
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert len(line) == 2
                edges.append([int(line[0]), int(line[1])])

        # 将边关系数据转换为 NumPy 数组
        adj = np.array(edges).transpose()

        # adj = sp.csr_matrix(adj)

        graph_node_features_dict = {}
        graph_labels_dict = {}

        if dataset_str == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(
                        line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(
                        line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(
                        line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(
                        line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        features_list = []
        for key in sorted(graph_node_features_dict):
            features_list.append(graph_node_features_dict[key])
        features = np.vstack(features_list)
        features = sp.csr_matrix(features)

        labels_list = []
        for key in sorted(graph_labels_dict):
            labels_list.append(graph_labels_dict[key])

        label_classes = max(labels_list) + 1
        labels = np.eye(label_classes)[labels_list]

        splits_file_path = 'splits/' + dataset_str + \
            '_split_0.6_0.2_' + str(split) + '.npz'

        with np.load(splits_file_path) as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']

        idx_train = np.where(train_mask == 1)[0]
        idx_val = np.where(val_mask == 1)[0]
        idx_test = np.where(test_mask == 1)[0]

    # adj = data_normalize(adj + sp.eye(adj.shape[0]))
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    features = data_normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels))[1]
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test



