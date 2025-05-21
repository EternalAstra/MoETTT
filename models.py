import torch
import torch.nn as nn
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv, MessagePassing , SGConv, GATConv, JumpingKnowledge, APPNP, GCN2Conv, SAGEConv
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter_min ,scatter_std
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np
import random
from tqdm import tqdm
from torch.nn.parameter import Parameter
from collections import deque
import scipy
import math
from torch_geometric.utils import (add_remaining_self_loops, homophily,
                                   remove_self_loops)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import scipy.sparse as sp
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from sklearn.preprocessing import normalize as sk_normalize
import torch.nn.init as init
from typing import Optional
from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)
from torch.distributions.normal import Normal

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, input_tensor=False):
        if not input_tensor:
            x = data.graph['node_feat']
        else:
            x = data
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, save_mem=False, use_bn=True):
        super(GCN, self).__init__()
        cached = False
        add_self_loops = True
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))
        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x = data.graph['node_feat']
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, data.graph['edge_index'])
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, data.graph['edge_index'])
        return x



class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, heads=2, add_self_loops=True):
        super(GAT, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, concat=True, add_self_loops=add_self_loops))

        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels*heads))
        for _ in range(num_layers - 2):

            self.convs.append(
                    GATConv(hidden_channels*heads, hidden_channels, heads=heads, concat=True, add_self_loops=add_self_loops) )
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels*heads))

        self.convs.append(
            GATConv(hidden_channels*heads, out_channels, heads=heads, concat=False, add_self_loops=add_self_loops))

        self.dropout = dropout
        self.activation = F.elu
        self.num_layers = num_layers

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


    def forward(self, data, adjs=None, x_batch=None):
        x = data.graph['node_feat']
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, data.graph['edge_index'])
            x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, data.graph['edge_index'])
        return x


    def inference(self, data, subgraph_loader):
        x_all = data.graph['node_feat']
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')
        total_edges = 0
        device = x_all.device
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = self.bns[i](x)
                    x = self.activation(x)
                xs.append(x.cpu())
                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


class H2GCNConv(nn.Module):
    def __init__(self):
        super(H2GCNConv, self).__init__()

    def reset_parameters(self):
        pass

    def forward(self, x, adj_t, adj_t2):
        x1 = matmul(adj_t, x)
        x2 = matmul(adj_t2, x)
        return torch.cat([x1, x2], dim=1)


class H2GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_index, num_nodes,
                 num_layers=2, dropout=0.5, num_mlp_layers=1,
                 use_bn=True, conv_dropout=True):
        super(H2GCN, self).__init__()

        #多层感知机（MLP），用于对输入特征进行嵌入
        self.feature_embed = MLP(in_channels, hidden_channels,hidden_channels, num_layers=num_mlp_layers, dropout=dropout)

        self.convs = nn.ModuleList()

        self.convs.append(H2GCNConv())

        self.bns = nn.ModuleList()

        self.bns.append(nn.BatchNorm1d(hidden_channels * 2 * len(self.convs)))

        for l in range(num_layers - 1):
            self.convs.append(H2GCNConv())
            if l != num_layers - 2:
                self.bns.append(nn.BatchNorm1d(hidden_channels * 2 * len(self.convs)))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.conv_dropout = conv_dropout  # dropout neighborhood aggregation steps

        #JumpingKnowledge的作用是融合不同层
        self.jump = JumpingKnowledge('cat')

        last_dim = hidden_channels * (2 ** (num_layers + 1) - 1)
        #线性层，用于将聚合的特征映射到输出类别
        self.final_project = nn.Linear(last_dim, out_channels)

        self.num_nodes = num_nodes
        #初始化邻接矩阵和二跳邻接矩阵
        self.init_adj(edge_index)

    def reset_parameters(self):
        self.feature_embed.reset_parameters()
        self.final_project.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def init_adj(self, edge_index):
        """ cache normalized adjacency and normalized strict two-hop adjacency,
        neither has self loops
        """
        n = self.num_nodes

        if isinstance(edge_index, SparseTensor):
            dev = edge_index.device
            adj_t = edge_index
            adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
            adj_t[adj_t > 0] = 1
            adj_t[adj_t < 0] = 0
            adj_t = SparseTensor.from_scipy(adj_t).to(dev)
        elif isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=None, sparse_sizes=(n, n))

        adj_t.remove_diag(0)
        adj_t2 = matmul(adj_t, adj_t)
        adj_t2.remove_diag(0)
        adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
        adj_t2 = scipy.sparse.csr_matrix(adj_t2.to_scipy())
        adj_t2 = adj_t2 - adj_t
        adj_t2[adj_t2 > 0] = 1
        adj_t2[adj_t2 < 0] = 0

        adj_t = SparseTensor.from_scipy(adj_t)
        adj_t2 = SparseTensor.from_scipy(adj_t2)

        adj_t = gcn_norm(adj_t, None, n, add_self_loops=False)
        adj_t2 = gcn_norm(adj_t2, None, n, add_self_loops=False)

        self.adj_t = adj_t.to(edge_index.device)
        self.adj_t2 = adj_t2.to(edge_index.device)

    def forward(self, data):
        x = data.graph['node_feat']
        n = data.graph['num_nodes']

        adj_t = self.adj_t
        adj_t2 = self.adj_t2

        #第一层的embedding是feature过一层MLP
        x = self.feature_embed(data)
        x = self.activation(x)
        xs = [x] #不同层embedding的集合


        #self.training是标志位，标记是否处于训练中
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, adj_t2)
            if self.use_bn:
                x = self.bns[i](x)
            xs.append(x)
            if self.conv_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t, adj_t2)

        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)

        xs.append(x)

        x = self.jump(xs)
        if not self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final_project(x)
        return x



class APPNP_Net(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dprate=.0, dropout=.5, K=10, alpha=.1, num_layers=3):
        super(APPNP_Net, self).__init__()

        self.mlp = MLP(in_channels, hidden_channels, out_channels, num_layers=num_layers, dropout=dropout)
        #k为迭代次数，alpha为传播率
        self.prop1 = APPNP(K, alpha)

        self.dprate = dprate
        self.dropout = dropout

    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, data):
        edge_index = data.graph['edge_index']
        x = self.mlp(data)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return x
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return x


class low_Conv(nn.Module):
    def __init__(self):
        super(low_Conv, self).__init__()

    def reset_parameters(self):
        pass

    def forward(self, adj, x):
        x = matmul(adj, x)
        return x

class high_Conv(nn.Module):
    def __init__(self):
        super(high_Conv, self).__init__()

    def reset_parameters(self):
        pass

    def forward(self, adj, x):
        # 计算度矩阵D和D的逆平方根
        D = torch.diag(torch.sum(adj, dim=1))
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.diag(D)))
        # 计算归一化的图拉普拉斯矩阵
        L_norm = torch.eye(adj.size(0)) - torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)
        # 使用归一化拉普拉斯矩阵作为高频滤波器
        x = torch.mm(L_norm, x)
        return x




class LINKX(nn.Module):
    """
    a = MLP_1(A), x = MLP_2(X), MLP_3(sigma(W_1[a, x] + a + x))
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_nodes, dropout=.5,
                 inner_activation=False, inner_dropout=False, init_layers_A=1, init_layers_X=1):
        super(LINKX, self).__init__()
        self.mlpA = MLP(num_nodes, hidden_channels, hidden_channels, init_layers_A, dropout=0)
        self.mlpX = MLP(in_channels, hidden_channels, hidden_channels, init_layers_X, dropout=0)
        self.W = nn.Linear(2 * hidden_channels, hidden_channels)
        self.mlp_final = MLP(hidden_channels, hidden_channels, out_channels, num_layers, dropout=dropout)
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.A = None
        self.inner_activation = inner_activation
        self.inner_dropout = inner_dropout

    def reset_parameters(self):
        self.mlpA.reset_parameters()
        self.mlpX.reset_parameters()
        self.W.reset_parameters()
        self.mlp_final.reset_parameters()

    def forward(self, data):
        m = data.graph['num_nodes']
        row, col = data.graph['edge_index']
        row = row - row.min()
        A = SparseTensor(row=row, col=col,sparse_sizes=(m, self.num_nodes)).to_torch_sparse_coo_tensor()
        xA = self.mlpA(A, input_tensor=True)
        xX = self.mlpX(data.graph['node_feat'], input_tensor=True)
        x = torch.cat((xA, xX), axis=-1)
        x = self.W(x)
        if self.inner_dropout:
            x = F.dropout(x)
        if self.inner_activation:
            x = F.relu(x)
        x = F.relu(x + xA + xX)
        x = self.mlp_final(x, input_tensor=True)
        return x


class LINK(nn.Module):
    """ logistic regression on adjacency matrix """

    def __init__(self, num_nodes, out_channels):
        super(LINK, self).__init__()
        self.W = nn.Linear(num_nodes, out_channels)
        self.num_nodes = num_nodes

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, data):
        N = data.graph['num_nodes']
        edge_index = data.graph['edge_index']
        if isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            row = row - row.min()  # for sampling
            A = SparseTensor(row=row, col=col, sparse_sizes=(N, self.num_nodes)).to_torch_sparse_coo_tensor()
        elif isinstance(edge_index, SparseTensor):
            A = edge_index.to_torch_sparse_coo_tensor()
        logits = self.W(A)
        return logits

class LINK_Concat(nn.Module):
    """ concate A and X as joint embeddings i.e. MLP([A;X])"""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_nodes, dropout=.5, cache=True):
        super(LINK_Concat, self).__init__()
        self.mlp = MLP(in_channels + num_nodes, hidden_channels, out_channels, num_layers, dropout=dropout)
        self.in_channels = in_channels
        self.cache = cache
        self.x = None

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, data):
        if (not self.cache) or (not isinstance(self.x, torch.Tensor)):
            N = data.graph['num_nodes']
            feat_dim = data.graph['node_feat']
            row, col = data.graph['edge_index']
            col = col + self.in_channels
            feat_nz = data.graph['node_feat'].nonzero(as_tuple=True)
            feat_row, feat_col = feat_nz
            full_row = torch.cat((feat_row, row))
            full_col = torch.cat((feat_col, col))
            value = data.graph['node_feat'][feat_nz]
            full_value = torch.cat((value,
                                    torch.ones(row.shape[0], device=value.device)))
            x = SparseTensor(row=full_row, col=full_col,
                             sparse_sizes=(N, N + self.in_channels)
                             ).to_torch_sparse_coo_tensor()
            if self.cache:
                self.x = x
        else:
            x = self.x
        logits = self.mlp(x, input_tensor=True)
        return logits


class SGC(nn.Module):
    def __init__(self, in_channels, out_channels, hops):
        """ takes 'hops' power of the normalized adjacency"""
        super(SGC, self).__init__()
        self.conv = SGConv(in_channels, out_channels, hops, cached=True)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, data):
        edge_index = data.graph['edge_index']
        x = data.graph['node_feat']
        x = self.conv(x, edge_index)
        return x


class SGCMem(nn.Module):
    def __init__(self, in_channels, out_channels, hops):
        """ lower memory version (if out_channels < in_channels)
        takes weight multiplication first, then propagate
        takes hops power of the normalized adjacency
        """
        super(SGCMem, self).__init__()

        self.lin = nn.Linear(in_channels, out_channels)
        self.hops = hops

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, data):
        edge_index = data.graph['edge_index']
        x = data.graph['node_feat']
        x = self.lin(x)
        n = data.graph['num_nodes']
        edge_weight = None

        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, n, False,
                dtype=x.dtype)
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False,
                dtype=x.dtype)
            edge_weight = None
            adj_t = edge_index

        for _ in range(self.hops):
            x = matmul(adj_t, x)

        return x


class MultiLP(nn.Module):
    """ label propagation, with possibly multiple hops of the adjacency """

    def __init__(self, out_channels, alpha, hops, num_iters=50, mult_bin=False):
        super(MultiLP, self).__init__()
        self.out_channels = out_channels
        self.alpha = alpha
        self.hops = hops
        self.num_iters = num_iters
        self.mult_bin = mult_bin  # handle multiple binary tasks

    def forward(self, data, train_idx):
        n = data.graph['num_nodes']
        edge_index = data.graph['edge_index']
        edge_weight = None

        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, n, False)
            row, col = edge_index
            # transposed if directed
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False)
            edge_weight = None
            adj_t = edge_index

        y = torch.zeros((n, self.out_channels)).to(adj_t.device())
        if data.label.shape[1] == 1:
            # make one hot
            y[train_idx] = F.one_hot(data.label[train_idx], self.out_channels).squeeze(1).to(y)
        elif self.mult_bin:
            y = torch.zeros((n, 2 * self.out_channels)).to(adj_t.device())
            for task in range(data.label.shape[1]):
                y[train_idx, 2 * task:2 * task + 2] = F.one_hot(data.label[train_idx, task], 2).to(y)
        else:
            y[train_idx] = data.label[train_idx].to(y.dtype)
        result = y.clone()
        for _ in range(self.num_iters):
            for _ in range(self.hops):
                result = matmul(adj_t, result)
            result *= self.alpha
            result += (1 - self.alpha) * y

        if self.mult_bin:
            output = torch.zeros((n, self.out_channels)).to(result.device)
            for task in range(data.label.shape[1]):
                output[:, task] = result[:, 2 * task + 1]
            result = output

        return result


class MixHopLayer(nn.Module):
    def __init__(self, in_channels, out_channels, hops=2):
        super(MixHopLayer, self).__init__()
        self.hops = hops
        self.lins = nn.ModuleList()
        for hop in range(self.hops + 1):
            lin = nn.Linear(in_channels, out_channels)
            self.lins.append(lin)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        #先过了一个线性层
        xs = [self.lins[0](x)]
        for j in range(1, self.hops + 1):
            x_j = self.lins[j](x)
            #不同跳数的结果，就是在其前面成多少个adj_t
            for hop in range(j):
                x_j = matmul(adj_t, x_j)
            xs += [x_j]
        return torch.cat(xs, dim=1)


class MixHop(nn.Module):
    """ our implementation of MixHop
    some assumptions: the powers of the adjacency are [0, 1, ..., hops],
        with every power in between
    each concatenated layer has the same dimension --- hidden_channels
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, hops=2):
        super(MixHop, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(MixHopLayer(in_channels, hidden_channels, hops=hops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels * (hops + 1)))
        for _ in range(num_layers - 2):
            self.convs.append(
                MixHopLayer(hidden_channels * (hops + 1), hidden_channels, hops=hops))
            self.bns.append(nn.BatchNorm1d(hidden_channels * (hops + 1)))

        self.convs.append(
            MixHopLayer(hidden_channels * (hops + 1), out_channels, hops=hops))

        # note: uses linear projection instead of paper's attention output
        self.final_project = nn.Linear(out_channels * (hops + 1), out_channels)

        self.dropout = dropout
        self.activation = F.relu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.final_project.reset_parameters()

    def forward(self, data):
        x = data.graph['node_feat']
        n = data.graph['num_nodes']
        edge_index = data.graph['edge_index']
        edge_weight = None
        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, n, False,
                dtype=x.dtype)
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False,
                dtype=x.dtype)
            edge_weight = None
            adj_t = edge_index

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x = self.final_project(x)
        return x


class GPR_prop(MessagePassing):
    '''
    GPRGNN, from original repo https://github.com/jianhao2016/GPRGNN
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        # 表示传播的最大步数或图卷积的最大阶数
        self.K = K
        # 一个字符串，指定初始化方法，可以是 'SGC', 'PPR', 'NPPR', 'Random', 或 'WS'。
        self.Init = Init
        # 在不同初始化方法中起不同作用，通常与传播或衰减系数相关。
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0 * np.ones(K + 1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
            TEMP[-1] = (1 - alpha) ** K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha) ** np.arange(K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3 / (K + 1))
            TEMP = np.random.uniform(-bound, bound, K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = nn.Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        nn.init.zeros_(self.temp)
        for k in range(self.K + 1):
            self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
        self.temp.data[-1] = (1 - self.alpha) ** self.K

    def forward(self, x, edge_index, edge_weight=None):
        # 标准化邻接矩阵
        if isinstance(edge_index, torch.Tensor):
            edge_index, norm = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
            norm = None
        # 初始化隐藏状态
        hidden = x * (self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k + 1]
            # 每次propagate乘以其权重系数
            hidden = hidden + gamma * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,self.temp)


class GPRGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, Init='Random', dprate=.0, dropout=.5, K=10, alpha=.1,Gamma=None, num_layers=3):
        super(GPRGNN, self).__init__()

        self.mlp = MLP(in_channels, hidden_channels, out_channels, num_layers=num_layers, dropout=dropout)
        self.prop1 = GPR_prop(K, alpha, Init, Gamma)

        self.Init = Init
        self.dprate = dprate
        self.dropout = dropout

    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, data):
        edge_index = data.graph['edge_index']
        x = self.mlp(data)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return x
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return x


class GCNII(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, alpha, theta, shared_weights=True,
                 dropout=0.5):
        super(GCNII, self).__init__()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.bns = nn.ModuleList()
        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x = data.graph['node_feat']
        n = data.graph['num_nodes']
        edge_index = data.graph['edge_index']
        edge_weight = None
        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, n, False, dtype=x.dtype)
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False, dtype=x.dtype)
            edge_weight = None
            adj_t = edge_index

        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for i, conv in enumerate(self.convs):
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = self.bns[i](x)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)
        return x


class HIGHGCNConv(MessagePassing):
    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            improved: bool = True,
            cached: bool = True,
            add_self_loops: bool = True,
            normalize: bool = True,
            bias: bool = True,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = Linear(in_channels, out_channels, bias=False,weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:



        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=(x.size(0), x.size(0)))

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out: Tensor, x: Tensor ) -> Tensor:
        # Restruct the update,to distinct the GCN
        return self.lin(x - aggr_out)


class HighPassConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(HighPassConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 3: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, norm=norm)

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
    def message(self, x_j, norm):
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out, x):
        # Step 5: Subtract the aggregated features from the original features
        return self.lin(x - aggr_out)



class HighPassGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(HighPassGCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(HighPassConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                HighPassConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(HighPassConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x = data.graph['node_feat']
        edge_index  = data.graph['edge_index']
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return torch.nn.functional.log_softmax(x, dim=-1)


class EdgeDiscriminator(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim):
        super(EdgeDiscriminator, self).__init__()
        self.layer1 = nn.Linear(2 * node_feat_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1)

    def forward(self, subgraph_pair, node_features):
        # 获取节点对的特征向量
        src_nodes = subgraph_pair[:, 0]
        tgt_nodes = subgraph_pair[:, 1]

        src_node_features = node_features[src_nodes]
        tgt_node_features = node_features[tgt_nodes]

        pair_features = torch.cat((src_node_features, tgt_node_features), dim=-1)

        hidden = F.relu(self.layer1(pair_features))
        confidences = torch.sigmoid(self.layer2(hidden)).squeeze()

        return confidences

    def __call__(self, subgraph_pair, node_features):
        return self.forward(subgraph_pair, node_features)

class GatingNetwork(torch.nn.Module):
    def __init__(self, node_feature_dim, hidden_channels, out_channels,max_nodes, mlp_layers):
        super(GatingNetwork, self).__init__()
        self.mlp1 = self._build_mlp(max_nodes * 2, hidden_channels, mlp_layers)
        self.mlp2 = self._build_mlp(max_nodes * 2, hidden_channels, mlp_layers)

        # 主MLP用于处理拼接后的特征
        self.main_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_channels, out_channels),
            torch.nn.Softmax(dim=-1)
        )


        self.edge_discriminator = EdgeDiscriminator(node_feat_dim=node_feature_dim, hidden_dim=16)
        self.all_subgraphs = None
        self.degreelist = None
        self.homo = None
        self.max_node = max_nodes
        self.num_nodes = None
    @staticmethod
    def _build_mlp(input_dim, hidden_dim, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.SiLU())
            input_dim = hidden_dim
        return torch.nn.Sequential(*layers)


    def forward(self, data):
        edge_index = data.graph['edge_index']
        edge_index, _ = remove_self_loops(edge_index)
        node_features = data.graph['node_feat']
        self.num_nodes = data.label.shape[0]
        row, col = edge_index[0], edge_index[1]

        # 第一步：通过随机游走的方式，获得中心节点的局部子图，上限为10个
        if self.all_subgraphs is None:
            # 第一步：通过随机游走的方式，获得中心节点的局部子图，上限为10个
            self.all_subgraphs = self.random_walk(row, col, self.num_nodes, max_nodes=self.max_node)

        if self.homo is None:
            self.homo = self.get_homophily(data)

        # 第二步：判断局部子图数组中每个节点和第一个节点连边的置信度

        edge_confidences = self.discriminate_edges(node_features, self.all_subgraphs)

        # 第三步：局部子图中每个节点的置信度和它本身的归一化的度进行拼接，得到中心节点的feature。
        features_list = self.aggregate_features(self.all_subgraphs, edge_confidences, edge_index)


        features_tensor = torch.stack(features_list)
        global_feature = torch.mean(features_tensor, dim=0).expand(self.num_nodes, -1)


        # 计算local_pattern和global_pattern
        local_patterns = self.mlp1(features_tensor)
        global_patterns = self.mlp2(global_feature)

        # 将local_patterns和global_patterns拼接，并通过主MLP和softmax
        mix_patterns = torch.cat([local_patterns, global_patterns], dim=-1)

        gating_scores = self.main_mlp(mix_patterns)


        return gating_scores

    def random_walk(self, row, col, num_nodes, max_nodes):
        all_subgraphs = []
        for start_node in range(num_nodes):
            visited = {start_node: 0}  # node: depth
            queue = deque([(start_node, 0, None)])  # node, depth, parent

            while queue and len(visited) < max_nodes:
                current_node, depth, parent = queue.popleft()
                neighbors = col[row == current_node].tolist()

                # Ensure all one-hop neighbors are visited
                if depth == 0:
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            visited[neighbor] = 1
                            queue.append((neighbor, 1, current_node))

                # Random walk step
                if neighbors:
                    # continue to a random neighbor
                    next_node = np.random.choice(neighbors)
                    next_depth = depth + 1

                    if next_node not in visited:
                        visited[next_node] = next_depth
                        queue.append((next_node, next_depth, current_node))

            # Sort nodes by depth
            sorted_nodes = sorted(visited.items(), key=lambda x: x[1])
            subgraph = [node for node, _ in sorted_nodes]
            # 如果子图节点数不足subgraph_size，进行填充
            if len(subgraph) < max_nodes:
                subgraph += [start_node] * (max_nodes - len(subgraph))
            # 如果子图节点多有于max_nodes 进行截断
            elif len(subgraph) > max_nodes:
                subgraph = subgraph[:max_nodes]


            all_subgraphs.append(subgraph)
        return all_subgraphs

    def discriminate_edges(self, node_features, all_subgraphs):

        # 将所有子图转换为张量列表，并转化为一个二维张量 (batch_size, subgraph_size)
        all_subgraphs_tensor = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(subgraph, dtype=torch.long) for subgraph in all_subgraphs],
            batch_first=True
        )

        # 获取子图的数量和每个子图中的节点数量
        batch_size, subgraph_size = all_subgraphs_tensor.size()

        # 生成每个subgraph的所有 pair，与第一个节点配对 (batch_size, subgraph_size, 2)
        first_nodes = all_subgraphs_tensor[:, 0].unsqueeze(1).expand(batch_size, subgraph_size).unsqueeze(2)
        subgraph_pairs = torch.stack((all_subgraphs_tensor.unsqueeze(2), first_nodes), dim=-1)

        # 将subgraph_pairs reshape成 (batch_size * subgraph_size, 2)
        subgraph_pairs_reshaped = subgraph_pairs.view(-1, 2)

        # 使用边判别器计算每个点对之间的置信度，得到 (batch_size * subgraph_size, 1)
        sub_edge_confidences = self.edge_discriminator(subgraph_pairs_reshaped, node_features)

        # 将结果重塑回 (batch_size, subgraph_size)
        edge_probabilities = sub_edge_confidences.view(batch_size, subgraph_size)

        return edge_probabilities


    def compute_degreelist(self, edge_index):
        # 根据edge_index计算每个节点的度
        num_nodes = edge_index.max().item() + 1  # 假设节点索引从0开始
        degreelist = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)

        # 计算每个节点的度
        degreelist.index_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.long, device=edge_index.device))
        degreelist.index_add_(0, edge_index[1], torch.ones(edge_index.size(1), dtype=torch.long, device=edge_index.device))


        return degreelist

    def aggregate_features(self, all_subgraphs, edge_confidences, edge_index):
        aggregated_feat = []

        if self.degreelist is None:
            self.degreelist = self.compute_degreelist(edge_index)


        for i, subgraph in enumerate(all_subgraphs):
            subgraph_nodes = torch.tensor(subgraph, dtype=torch.long)
            # 根据degreelist得到subgraph这个一维张量中每个节点对应的度
            degrees = self.degreelist[subgraph_nodes].to(device)

            max_degree = degrees.max().float()
            if max_degree == 0:
                degrees = degrees.float()
            else:
                degrees = degrees.float() / max_degree
            # 将edge_confidences[i]和degrees两个一维张量进行拼接，得到一个长度两倍的一维张量
            merge_feat = torch.cat([edge_confidences[i].unsqueeze(1), degrees.float().unsqueeze(1)], dim=-1)
            #将二维tensor merge_feat变成一维tensor
            merge_feat = merge_feat.view(-1)
            aggregated_feat.append(merge_feat)

        return aggregated_feat

    def get_embed(self,feat,edge_index):
        row, col = edge_index[0], edge_index[1]
        all_subgraphs = self.random_walk(row, col, self.num_nodes, max_nodes=self.max_node)
        edge_confidences = self.discriminate_edges(feat, all_subgraphs)
        features_list = self.aggregate_features(all_subgraphs, edge_confidences, edge_index)
        features_tensor = torch.stack(features_list)
        return features_tensor
    def reset_parameters(self):
        # 重置四个MLP的参数
        self.mlp1.apply(self.reset_mlp)
        self.mlp2.apply(self.reset_mlp)
        self.main_mlp.apply(self.reset_mlp)

    @staticmethod
    def reset_mlp(m):
        if isinstance(m, torch.nn.Linear):
            m.reset_parameters()


    def get_homophily(self,data):
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
        return results



class SSLHead(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=128):
        super(SSLHead, self).__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
class MoEGCN(torch.nn.Module):
    def __init__(self,in_channels , hidden_channels , out_channels, num_layers, dropout, num_nodes, dataset, max_node, mlp_layer,name,run):
        super(MoEGCN, self).__init__()
        # 加载专家模型
        # model1_path = f'./models/{name}/{run}/gcn.pt'
        # model2_path = f'./models/{name}/{run}/highpassgcn.pt'
        # model3_path = f'./models/{name}/{run}/mlp.pt'
        # model4_path = f'./models/{name}/{run}/lsgnnhigh.pt'
        # model5_path = f'./models/{name}/{run}/lsgnnlow.pt'
        #
        # model1_path = f'./models/{name}/{run}/gcn_part.pt'
        # model2_path = f'./models/{name}/{run}/highpassgcn_part.pt'
        # model3_path = f'./models/{name}/{run}/mlp_part.pt'
        # model4_path = f'./models/{name}/{run}/lsgnnhigh_part.pt'
        # model5_path = f'./models/{name}/{run}/lsgnnlow_part.pt'

        # self.plot = 0
        # self.expert1 = torch.load(model1_path)
        # self.expert2 = torch.load(model2_path)
        # self.expert3 = torch.load(model3_path)
        # self.expert4 = torch.load(model4_path)
        # self.expert5 = torch.load(model5_path)

        self.expert1 = GCN(in_channels, hidden_channels, out_channels, num_layers, dropout)
        self.expert2 = HighPassGCN(in_channels, hidden_channels, out_channels, num_layers, dropout)
        self.expert3 = MLP(in_channels, hidden_channels, out_channels, num_layers, dropout)
        self.expert4 = LSGNN_HIGH(in_channels, hidden_channels, out_channels, num_nodes)
        self.expert5 = LSGNN_LOW(in_channels, hidden_channels, out_channels, num_nodes)

        # 冻结专家模型的参数
        # self._freeze_experts([self.expert1, self.expert2, self.expert3, self.expert4, self.expert5])

        # 定义门控网络
        self.gating_network = GatingNetwork(dataset.graph['node_feat'].shape[1],hidden_channels,5,max_node, mlp_layer)

    def _freeze_experts(self, experts):
        for expert in experts:
            for param in expert.parameters():
                param.requires_grad = False

    def forward(self, data):
        # 获取每个专家的预测
        out1 = self.expert1(data)
        out2 = self.expert2(data)
        out3 = self.expert3(data)
        out4 = self.expert4(data)
        out5 = self.expert5(data)


        # 将所有专家的输出堆叠成一个张量[5, num_nodes, output_dim]
        experts_output = torch.stack([out1, out2, out3, out4, out5],dim=0)

        # 通过门控网络获取权重
        gating_scores_list = self.gating_network(data)



        # 使用 gating_scores_list 对专家的输出进行加权求和
        gating_scores = gating_scores_list.t().unsqueeze(-1)  #  [5, num_nodes, 1]
        out = torch.sum(experts_output * gating_scores, dim=0)  #  [num_nodes, output_dim]
        return out

    def reset_parameters(self):
        self.gating_network.reset_parameters()

    def plot_weight_distribute(self,data,gating_scores_list):

        homophily_scores = self.get_homophily(data)  # 获取节点的 homophily

        # 按照 homophily 得分对节点排序
        sorted_indices = torch.argsort(homophily_scores)

        # 将节点分成五组
        num_groups = 5
        size_group = len(sorted_indices) // num_groups
        grouped_indices = [sorted_indices[i * size_group: (i + 1) * size_group] for i in range(num_groups)]

        average_gating_scores = []

        for group in grouped_indices:
            group_scores = gating_scores_list[group]  # 获取该组的 gating scores
            average_scores = group_scores.mean(dim=0).tolist()  # 计算该组每个专家的平均权重
            average_gating_scores.append(average_scores)

        average_gating_scores = np.array(average_gating_scores)
        print(average_gating_scores)
        # 绘制柱状图
        fig, ax = plt.subplots()
        width = 0.1  # 柱状图每个柱子的宽度

        x = np.arange(num_groups)

        for i in range(average_gating_scores.shape[1]):
            ax.bar(x + i * width, average_gating_scores[:, i], width, label=f'Expert {i + 1}')

        ax.set_xlabel('Homophily Group')
        ax.set_ylabel('Average Expert Weights')
        ax.set_title('Average Expert Weights by Homophily Group')
        ax.legend()
        plt.show()

    def get_homophily(self,data):
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
        return results



class MoETTT(torch.nn.Module):
    def __init__(self,in_channels , hidden_channels , out_channels, num_layers, dropout,num_nodes, dataset, max_node, mlp_layer,name,run):
        super(MoETTT, self).__init__()

        self.expert_low = GCN(in_channels, hidden_channels, out_channels, num_layers, dropout)
        self.expert_high = HighPassGCN(in_channels, hidden_channels, out_channels, num_layers, dropout)
        self.expert_mid = MLP(in_channels, hidden_channels, out_channels, num_layers, dropout)

        # 冻结专家模型的参数
        # self._freeze_experts([self.expert1, self.expert2, self.expert3, self.expert4, self.expert5])

        # 定义门控网络
        self.gating_network = GatingNetwork(dataset.graph['node_feat'].shape[1],hidden_channels,3,max_node, mlp_layer)
        #可学习的聚类中心
        self.routing_centers = nn.Parameter(torch.randn(3, 128), requires_grad=True)


        # self.ssl_head = SSLHead(in_dim=hidden_channels)
    def _freeze_experts(self, experts):
        for expert in experts:
            for param in expert.parameters():
                param.requires_grad = False

    def forward(self, data):
        # 获得每个专家的输出
        out_low = self.expert_low(data)  # [num_nodes, out_dim]
        out_high = self.expert_high(data)  # [num_nodes, out_dim]
        out_mid = self.expert_mid(data)  # [num_nodes, out_dim]

        experts_output = torch.stack([out_low, out_high, out_mid], dim=0)

        # 通过门控网络获取权重
        gating_scores_list = self.gating_network(data)

        # 使用 gating_scores_list 对专家的输出进行加权求和
        gating_scores = gating_scores_list.t().unsqueeze(-1)  #  [5, num_nodes, 1]
        out = torch.sum(experts_output * gating_scores, dim=0)  #  [num_nodes, output_dim]
        return out



    def reset_parameters(self):
        self.gating_network.reset_parameters()

    def plot_weight_distribute(self,data,gating_scores_list):

        homophily_scores = self.get_homophily(data)  # 获取节点的 homophily

        # 按照 homophily 得分对节点排序
        sorted_indices = torch.argsort(homophily_scores)

        # 将节点分成五组
        num_groups = 5
        size_group = len(sorted_indices) // num_groups
        grouped_indices = [sorted_indices[i * size_group: (i + 1) * size_group] for i in range(num_groups)]

        average_gating_scores = []

        for group in grouped_indices:
            group_scores = gating_scores_list[group]  # 获取该组的 gating scores
            average_scores = group_scores.mean(dim=0).tolist()  # 计算该组每个专家的平均权重
            average_gating_scores.append(average_scores)

        average_gating_scores = np.array(average_gating_scores)
        print(average_gating_scores)
        # 绘制柱状图
        fig, ax = plt.subplots()
        width = 0.1  # 柱状图每个柱子的宽度

        x = np.arange(num_groups)

        for i in range(average_gating_scores.shape[1]):
            ax.bar(x + i * width, average_gating_scores[:, i], width, label=f'Expert {i + 1}')

        ax.set_xlabel('Homophily Group')
        ax.set_ylabel('Average Expert Weights')
        ax.set_title('Average Expert Weights by Homophily Group')
        ax.legend()
        plt.show()

    def get_homophily(self,data):
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
        return results







def scipy_coo_matrix_to_torch_sparse_tensor(sparse_mx):
    indices1 = torch.from_numpy(np.stack([sparse_mx.row, sparse_mx.col]).astype(np.int64))
    values1 = torch.from_numpy(sparse_mx.data)
    shape1 = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices=indices1, values=values1, size=shape1)

def cal_filter(edge_index, num_nodes):
    edge_index = edge_index.cpu()
    N = num_nodes

    # A
    edge_index, _ = remove_self_loops(edge_index=edge_index)
    edge_index_sl, _ = add_remaining_self_loops(edge_index=edge_index)

    # D
    adj_data = np.ones([edge_index.shape[1]], dtype=np.float32)
    adj_sp = sp.csr_matrix((adj_data, (edge_index[0], edge_index[1])), shape=[N, N])

    adj_sl_data = np.ones([edge_index_sl.shape[1]], dtype=np.float32)
    adj_sl_sp = sp.csr_matrix((adj_sl_data, (edge_index_sl[0], edge_index_sl[1])), shape=[N, N])

    # D-1/2
    deg = np.array(adj_sl_sp.sum(axis=1)).flatten()
    deg_sqrt_inv = np.power(deg, -0.5)
    deg_sqrt_inv[deg_sqrt_inv == float('inf')] = 0.0
    deg_sqrt_inv = sp.diags(deg_sqrt_inv)

    # filters
    DAD = sp.coo_matrix(deg_sqrt_inv * adj_sp * deg_sqrt_inv)
    DAD = scipy_coo_matrix_to_torch_sparse_tensor(DAD)

    return DAD

class LSGNN(MessagePassing):
    """local similarity graph neural network"""

    def __init__(self, in_channels,hidden_channels , out_channels, num_nodes ,num_reduce_layers = 1,K = 5 ,beta = 1, gamma = 0.5,method = 'norm2',dropout= 0.5 ,A_embed = False ,out_mlp = False):
        super(LSGNN, self).__init__(aggr='add')

        hidden_channels = hidden_channels
        num_reduce_layers = num_reduce_layers

        self.K = K
        self.beta = beta
        self.gamma = gamma
        self.method = method
        self.dp = dropout


        self.dist_mlp = nn.Sequential(
            nn.Linear(2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1)
        )

        self.alpha_mlp = nn.Sequential(
            nn.Linear(2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 3 * self.K)
        )

        if num_reduce_layers == 1:
            self.reduce = [Parameter(torch.zeros([2 * self.K + 1, in_channels, hidden_channels]))]
        elif num_reduce_layers > 1:
            self.reduce = [Parameter(torch.zeros([2 * self.K + 1, in_channels, 2 * hidden_channels]))]
            for _ in range(num_reduce_layers - 2):
                self.reduce.append(Parameter(torch.zeros([2 * self.K + 1, 2 * hidden_channels, 2 * hidden_channels])))
            self.reduce.append(Parameter(torch.zeros([2 * self.K + 1, in_channels, hidden_channels])))
        else:
            raise NotImplementedError
        self.reset_parameters()
        self.A_embed = A_embed

        if self.A_embed:
            self.A_mlp = nn.Sequential(
                nn.Linear(num_nodes, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
            )
            final_nz = self.K + 2
        else:
            self.A_mlp = None
            final_nz = self.K + 1

        self.out_mlp = out_mlp
        if self.out_mlp:
            self.out_linear = nn.Sequential(
                nn.Linear(final_nz * hidden_channels, 2 * hidden_channels),
                nn.BatchNorm1d(2 * hidden_channels),
                nn.ReLU(),
                nn.Linear(2 * hidden_channels, out_channels)
            )
        else:
            self.out_linear = nn.Linear(final_nz * hidden_channels, out_channels)

        self.cache = None

    @torch.no_grad()
    def reset_parameters(self):
        for i, param in enumerate(self.reduce):
            self.register_parameter(f'reduce_{i}', param)
            nn.init.xavier_uniform_(param.data)

    def dist(self, data):
        edge_index = data.graph['edge_index']
        x = data.graph['node_feat']
        src, tgt = edge_index
        if self.method == 'cos':
            dist = (x[src] * x[tgt]).sum(dim=-1)
        elif self.method == 'norm2':
            dist = torch.norm(x[src] - x[tgt], p=2, dim=-1)
        dist = dist.view(-1, 1)
        dist = torch.cat([dist, dist.square()], dim=-1)
        return dist

    def local_sim(self, data, dist=None):
        edge_index = data.graph['edge_index']
        x = data.graph['node_feat']
        if dist is None:
            dist = self.dist(x, edge_index)
        _, tgt = edge_index
        dist = self.dist_mlp(dist).view(-1)
        return scatter_mean(dist, tgt, out=torch.zeros([x.shape[0]], device=x.device))

    def prop(self, data):
        x = data.graph['node_feat']
        edge_index = data.graph['edge_index']
        N, _ = x.shape
        dev = x.device

        DAD = cal_filter(edge_index=edge_index, num_nodes=N).to(dev)

        # beta
        I = torch.eye(N, device=dev)  # 创建一个阶数为n的单位矩阵
        filter_l = SparseTensor.from_dense(self.beta * I + DAD)  # 低通滤波矩阵
        filter_h = SparseTensor.from_dense((1 - self.beta) * I - DAD)  # 高通滤波矩阵


        # propagate first
        x = x.type(torch.float32)
        # first propagation
        x_L = x.clone()
        x_H = x.clone()
        x_L = self.propagate(edge_index=filter_l, x=x_L)
        x_H = self.propagate(edge_index=filter_h, x=x_H)
        out_L = [x_L]
        out_H = [x_H]
        x_L_sum = 0
        x_H_sum = 0

        # continue propagation
        for _ in range(1, self.K):
            x_L_sum = x_L_sum + out_L[-1]
            x_H_sum = x_H_sum + out_H[-1]
            x_L = self.propagate(
                edge_index=filter_l, x=(1 - self.gamma) * x - self.gamma * x_L_sum)
            x_H = self.propagate(
                edge_index=filter_h, x=(1 - self.gamma) * x - self.gamma * x_H_sum)

            out_L.append(x_L)
            out_H.append(x_H)

        x_out_L_out_H = torch.stack([x] + out_L + out_H, dim=0)

        return x_out_L_out_H

    def forward(self, data):
        edge_index = data.graph['edge_index']
        x = data.graph['node_feat']
        N, _ = x.shape
        dev = x.device
        # 这个forward函数中计算的dist、x_out_L_out_H会使计算变慢
        dist = self.dist(data)
        x_out_L_out_H = self.prop(data)
        # cal node sim
        local_sim = self.local_sim(data, dist)
        ls_ls2 = torch.cat([local_sim.view(-1, 1), local_sim.view(-1, 1).square()], dim=-1)

        # cal alpha
        alpha = self.alpha_mlp(ls_ls2)  # (N, 3K)
        stack_alpha = alpha.reshape([N, self.K, 3])
        alpha_I = stack_alpha[:, :, 0].t().unsqueeze(-1)
        alpha_L = stack_alpha[:, :, 1].t().unsqueeze(-1)
        alpha_H = stack_alpha[:, :, 2].t().unsqueeze(-1)

        # reduce dimensional
        for reduce_layer in self.reduce:
            x_out_L_out_H = torch.bmm(x_out_L_out_H, reduce_layer)
            x_out_L_out_H = F.normalize(x_out_L_out_H, p=2, dim=-1)
            x_out_L_out_H = F.relu(x_out_L_out_H)

        x = x_out_L_out_H[0, :, :]  # (N, hdim)
        out_I = x.expand(self.K, -1, -1)  # (K, N, hdim)
        out_L = x_out_L_out_H[1:self.K + 1, :, :]  # (K, N, hdim)
        out_H = x_out_L_out_H[self.K + 1:, :, :]  # (K, N, hdim)

        # fusion: (K, N, hdim)
        out = alpha_I * out_I + alpha_L * out_L + alpha_H * out_H

        # embedding A and concat representations
        if self.A_mlp is not None:
            A = SparseTensor(
                row=edge_index[0], col=edge_index[1],
                value=torch.ones([edge_index.shape[1]]).to(dev),
                sparse_sizes=[N, N]).to_torch_sparse_coo_tensor()
            A = self.A_mlp(A)
            out = torch.cat([x.unsqueeze(0), out, A.unsqueeze(0)], dim=0)
        else:
            out = torch.cat([x.unsqueeze(0), out], dim=0)

        # norm (K+1, N, hdim)
        out = F.normalize(out, p=2, dim=-1)
        out = F.dropout(out, self.dp, self.training)
        out = out.permute(1, 0, 2).reshape(N, -1)

        # prediction
        out = self.out_linear(out)
        return out

    def message(self, x_j, norm):
        # x_j: (E, out_channels)
        # norm: (E)
        return norm.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)


class LSGNN_HIGH(MessagePassing):
    """local similarity graph neural network"""

    def __init__(self, in_channels,hidden_channels , out_channels, num_nodes ,num_reduce_layers = 1,K = 5 ,beta = 1, gamma = 0.5,method = 'norm2',dropout= 0.5 ,A_embed = False ,out_mlp = False):
        super(LSGNN_HIGH, self).__init__(aggr='add')

        hidden_channels = hidden_channels
        num_reduce_layers = num_reduce_layers

        self.K = K
        self.beta = beta
        self.gamma = gamma
        self.method = method
        self.dp = dropout

        if num_reduce_layers == 1:
            self.reduce = [Parameter(torch.zeros([2 * self.K + 1, in_channels, hidden_channels]))]
        elif num_reduce_layers > 1:
            self.reduce = [Parameter(torch.zeros([2 * self.K + 1, in_channels, 2 * hidden_channels]))]
            for _ in range(num_reduce_layers - 2):
                self.reduce.append(Parameter(torch.zeros([2 * self.K + 1, 2 * hidden_channels, 2 * hidden_channels])))
            self.reduce.append(Parameter(torch.zeros([2 * self.K + 1, in_channels, hidden_channels])))
        else:
            raise NotImplementedError
        self.reset_parameters()
        self.A_embed = A_embed

        if self.A_embed:
            self.A_mlp = nn.Sequential(
                nn.Linear(num_nodes, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
            )
            final_nz = self.K + 2
        else:
            self.A_mlp = None
            final_nz = self.K + 1

        self.out_mlp = out_mlp
        if self.out_mlp:
            self.out_linear = nn.Sequential(
                nn.Linear(final_nz * hidden_channels, 2 * hidden_channels),
                nn.BatchNorm1d(2 * hidden_channels),
                nn.ReLU(),
                nn.Linear(2 * hidden_channels, out_channels)
            )
        else:
            self.out_linear = nn.Linear(final_nz * hidden_channels, out_channels)

        self.cache = None

    @torch.no_grad()
    def reset_parameters(self):
        for i, param in enumerate(self.reduce):
            self.register_parameter(f'reduce_{i}', param)
            nn.init.xavier_uniform_(param.data)


    def prop(self, data):
        x = data.graph['node_feat']
        edge_index = data.graph['edge_index']
        N, _ = x.shape
        dev = x.device

        DAD = cal_filter(edge_index=edge_index, num_nodes=N).to(dev)

        # beta
        I = torch.eye(N, device=dev)  # 创建一个阶数为n的单位矩阵
        filter_l = SparseTensor.from_dense(self.beta * I + DAD)  # 低通滤波矩阵
        filter_h = SparseTensor.from_dense((1 - self.beta) * I - DAD)  # 高通滤波矩阵
        # propagate first
        x = x.type(torch.float32)
        # first propagation
        x_L = x.clone()
        x_H = x.clone()
        x_L = self.propagate(edge_index=filter_l, x=x_L)
        x_H = self.propagate(edge_index=filter_h, x=x_H)
        out_L = [x_L]
        out_H = [x_H]
        x_L_sum = 0
        x_H_sum = 0
        # continue propagation
        for _ in range(1, self.K):
            x_L_sum = x_L_sum + out_L[-1]
            x_H_sum = x_H_sum + out_H[-1]
            x_L = self.propagate(
                edge_index=filter_l, x=(1 - self.gamma) * x - self.gamma * x_L_sum)
            x_H = self.propagate(
                edge_index=filter_h, x=(1 - self.gamma) * x - self.gamma * x_H_sum)
            out_L.append(x_L)
            out_H.append(x_H)

        x_out_L_out_H = torch.stack([x] + out_L + out_H, dim=0)

        return x_out_L_out_H

    def forward(self, data):
        edge_index = data.graph['edge_index']
        x = data.graph['node_feat']
        N, _ = x.shape
        dev = x.device

        x_out_L_out_H = self.prop(data)

        # reduce dimensional
        for reduce_layer in self.reduce:
            x_out_L_out_H = torch.bmm(x_out_L_out_H, reduce_layer)
            x_out_L_out_H = F.normalize(x_out_L_out_H, p=2, dim=-1)
            x_out_L_out_H = F.relu(x_out_L_out_H)

        x = x_out_L_out_H[0, :, :]  # (N, hdim)
        out = x_out_L_out_H[self.K + 1:, :, :]  # (K, N, hdim)



        # embedding A and concat representations
        if self.A_mlp is not None:
            A = SparseTensor(
                row=edge_index[0], col=edge_index[1],
                value=torch.ones([edge_index.shape[1]]).to(dev),
                sparse_sizes=[N, N]).to_torch_sparse_coo_tensor()
            A = self.A_mlp(A)
            out = torch.cat([x.unsqueeze(0), out, A.unsqueeze(0)], dim=0)
        else:
            out = torch.cat([x.unsqueeze(0), out], dim=0)

        # norm (K+1, N, hdim)
        out = F.normalize(out, p=2, dim=-1)
        out = F.dropout(out, self.dp, self.training)
        out = out.permute(1, 0, 2).reshape(N, -1)

        # prediction
        out = self.out_linear(out)
        return out

    def message(self, x_j, norm):
        # x_j: (E, out_channels)
        # norm: (E)
        return norm.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)
class LSGNN_LOW(MessagePassing):
    """local similarity graph neural network"""

    def __init__(self, in_channels,hidden_channels , out_channels, num_nodes ,num_reduce_layers = 1,K = 5 ,beta = 1, gamma = 0.5,method = 'norm2',dropout= 0.5 ,A_embed = False ,out_mlp = False):
        super(LSGNN_LOW, self).__init__(aggr='add')

        hidden_channels = hidden_channels
        num_reduce_layers = num_reduce_layers

        self.K = K
        self.beta = beta
        self.gamma = gamma
        self.method = method
        self.dp = dropout

        if num_reduce_layers == 1:
            self.reduce = [Parameter(torch.zeros([2 * self.K + 1, in_channels, hidden_channels]))]
        elif num_reduce_layers > 1:
            self.reduce = [Parameter(torch.zeros([2 * self.K + 1, in_channels, 2 * hidden_channels]))]
            for _ in range(num_reduce_layers - 2):
                self.reduce.append(Parameter(torch.zeros([2 * self.K + 1, 2 * hidden_channels, 2 * hidden_channels])))
            self.reduce.append(Parameter(torch.zeros([2 * self.K + 1, in_channels, hidden_channels])))
        else:
            raise NotImplementedError
        self.reset_parameters()
        self.A_embed = A_embed

        if self.A_embed:
            self.A_mlp = nn.Sequential(
                nn.Linear(num_nodes, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
            )
            final_nz = self.K + 2
        else:
            self.A_mlp = None
            final_nz = self.K + 1

        self.out_mlp = out_mlp
        if self.out_mlp:
            self.out_linear = nn.Sequential(
                nn.Linear(final_nz * hidden_channels, 2 * hidden_channels),
                nn.BatchNorm1d(2 * hidden_channels),
                nn.ReLU(),
                nn.Linear(2 * hidden_channels, out_channels)
            )
        else:
            self.out_linear = nn.Linear(final_nz * hidden_channels, out_channels)

        self.cache = None

    @torch.no_grad()
    def reset_parameters(self):
        for i, param in enumerate(self.reduce):
            self.register_parameter(f'reduce_{i}', param)
            nn.init.xavier_uniform_(param.data)


    def prop(self, data):
        x = data.graph['node_feat']
        edge_index = data.graph['edge_index']
        N, _ = x.shape
        dev = x.device

        DAD = cal_filter(edge_index=edge_index, num_nodes=N).to(dev)

        # beta
        I = torch.eye(N, device=dev)  # 创建一个阶数为n的单位矩阵
        filter_l = SparseTensor.from_dense(self.beta * I + DAD)  # 低通滤波矩阵
        filter_h = SparseTensor.from_dense((1 - self.beta) * I - DAD)  # 高通滤波矩阵
        # propagate first
        x = x.type(torch.float32)
        # first propagation
        x_L = x.clone()
        x_H = x.clone()
        x_L = self.propagate(edge_index=filter_l, x=x_L)
        x_H = self.propagate(edge_index=filter_h, x=x_H)
        out_L = [x_L]
        out_H = [x_H]
        x_L_sum = 0
        x_H_sum = 0
        # continue propagation
        for _ in range(1, self.K):
            x_L_sum = x_L_sum + out_L[-1]
            x_H_sum = x_H_sum + out_H[-1]
            x_L = self.propagate(
                edge_index=filter_l, x=(1 - self.gamma) * x - self.gamma * x_L_sum)
            x_H = self.propagate(
                edge_index=filter_h, x=(1 - self.gamma) * x - self.gamma * x_H_sum)
            out_L.append(x_L)
            out_H.append(x_H)

        x_out_L_out_H = torch.stack([x] + out_L + out_H, dim=0)

        return x_out_L_out_H

    def forward(self, data):
        edge_index = data.graph['edge_index']
        x = data.graph['node_feat']
        N, _ = x.shape
        dev = x.device

        x_out_L_out_H = self.prop(data)

        # reduce dimensional
        for reduce_layer in self.reduce:
            x_out_L_out_H = torch.bmm(x_out_L_out_H, reduce_layer)
            x_out_L_out_H = F.normalize(x_out_L_out_H, p=2, dim=-1)
            x_out_L_out_H = F.relu(x_out_L_out_H)

        x = x_out_L_out_H[0, :, :]  # (N, hdim)
        out = x_out_L_out_H[1:self.K + 1, :, :]  # (K, N, hdim)



        # embedding A and concat representations
        if self.A_mlp is not None:
            A = SparseTensor(
                row=edge_index[0], col=edge_index[1],
                value=torch.ones([edge_index.shape[1]]).to(dev),
                sparse_sizes=[N, N]).to_torch_sparse_coo_tensor()
            A = self.A_mlp(A)
            out = torch.cat([x.unsqueeze(0), out, A.unsqueeze(0)], dim=0)
        else:
            out = torch.cat([x.unsqueeze(0), out], dim=0)

        # norm (K+1, N, hdim)
        out = F.normalize(out, p=2, dim=-1)
        out = F.dropout(out, self.dp, self.training)
        out = out.permute(1, 0, 2).reshape(N, -1)

        # prediction
        out = self.out_linear(out)
        return out

    def message(self, x_j, norm):
        # x_j: (E, out_channels)
        # norm: (E)
        return norm.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)



class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, model_type, output_layer=0, variant=False):
        super(GraphConvolution, self).__init__()
        self.in_features, self.out_features, self.output_layer, self.model_type, self.variant = in_features, out_features, output_layer, model_type, variant
        self.att_low, self.att_high, self.att_mlp = 0, 0, 0
        if torch.cuda.is_available():
            self.weight_low, self.weight_high, self.weight_mlp = Parameter(torch.FloatTensor(in_features, out_features).cuda()), Parameter(
                torch.FloatTensor(in_features, out_features).cuda()), Parameter(torch.FloatTensor(in_features, out_features).cuda())
            self.att_vec_low, self.att_vec_high, self.att_vec_mlp = Parameter(torch.FloatTensor(out_features, 1).cuda(
            )), Parameter(torch.FloatTensor(out_features, 1).cuda()), Parameter(torch.FloatTensor(out_features, 1).cuda())
            self.low_param, self.high_param, self.mlp_param = Parameter(torch.FloatTensor(1, 1).cuda(
            )), Parameter(torch.FloatTensor(1, 1).cuda()), Parameter(torch.FloatTensor(1, 1).cuda())

            self.att_vec = Parameter(torch.FloatTensor(3, 3).cuda())

        else:
            self.weight_low, self.weight_high, self.weight_mlp = Parameter(torch.FloatTensor(in_features, out_features)), Parameter(
                torch.FloatTensor(in_features, out_features)), Parameter(torch.FloatTensor(in_features, out_features))
            self.att_vec_low, self.att_vec_high, self.att_vec_mlp = Parameter(torch.FloatTensor(out_features, 1)), Parameter(
                torch.FloatTensor(out_features, 1)), Parameter(torch.FloatTensor(out_features, 1))
            self.low_param, self.high_param, self.mlp_param = Parameter(torch.FloatTensor(
                1, 1)), Parameter(torch.FloatTensor(1, 1)), Parameter(torch.FloatTensor(1, 1))

            self.att_vec = Parameter(torch.FloatTensor(3, 3))
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight_mlp.size(1))
        std_att = 1. / math.sqrt(self.att_vec_mlp.size(1))

        std_att_vec = 1. / math.sqrt(self.att_vec.size(1))
        self.weight_low.data.uniform_(-stdv, stdv)
        self.weight_high.data.uniform_(-stdv, stdv)
        self.weight_mlp.data.uniform_(-stdv, stdv)
        self.att_vec_high.data.uniform_(-std_att, std_att)
        self.att_vec_low.data.uniform_(-std_att, std_att)
        self.att_vec_mlp.data.uniform_(-std_att, std_att)

        self.att_vec.data.uniform_(-std_att_vec, std_att_vec)

    def attention(self, output_low, output_high, output_mlp):
        T = 3
        att = torch.softmax(torch.mm(torch.sigmoid(torch.cat([torch.mm((output_low), self.att_vec_low), torch.mm(
            (output_high), self.att_vec_high), torch.mm((output_mlp), self.att_vec_mlp)], 1)), self.att_vec)/T, 1)
        return att[:, 0][:, None], att[:, 1][:, None], att[:, 2][:, None]

    def plot_weight_distribute(self,data,gating_scores_list):

        homophily_scores = self.get_homophily(data)  # 获取节点的 homophily

        # 按照 homophily 得分对节点排序
        sorted_indices = torch.argsort(homophily_scores)

        # 将节点分成五组
        num_groups = 5
        size_group = len(sorted_indices) // num_groups
        grouped_indices = [sorted_indices[i * size_group: (i + 1) * size_group] for i in range(num_groups)]

        average_gating_scores = []

        for group in grouped_indices:
            group_scores = gating_scores_list[group]  # 获取该组的 gating scores
            average_scores = group_scores.mean(dim=0).tolist()  # 计算该组每个专家的平均权重
            average_gating_scores.append(average_scores)

        average_gating_scores = np.array(average_gating_scores)

        # 绘制柱状图
        fig, ax = plt.subplots()
        width = 0.1  # 柱状图每个柱子的宽度

        x = np.arange(num_groups)

        for i in range(average_gating_scores.shape[1]):
            ax.bar(x + i * width, average_gating_scores[:, i], width, label=f'Expert {i + 1}')

        ax.set_xlabel('Homophily Group')
        ax.set_ylabel('Average Expert Weights')
        ax.set_title('Average Expert Weights by Homophily Group')
        ax.legend()
        plt.show()

    def get_homophily(self,data):
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
        return results

    def forward(self, input, adj_low, adj_high ,data):
        output = 0
        if self.model_type == 'mlp':
            output_mlp = (torch.mm(input, self.weight_mlp))
            return output_mlp
        elif self.model_type == 'sgc' or self.model_type == 'gcn':
            output_low = torch.mm(adj_low, torch.mm(input, self.weight_low))
            return output_low
        elif self.model_type == 'acmgcn' or self.model_type == 'acmsnowball':
            if self.variant:
                output_low = (torch.spmm(adj_low, F.relu(
                    torch.mm(input, self.weight_low))))
                output_high = (torch.spmm(adj_high, F.relu(
                    torch.mm(input, self.weight_high))))
                output_mlp = F.relu(torch.mm(input, self.weight_mlp))
            else:
                output_low = F.relu(torch.spmm(
                    adj_low, (torch.mm(input, self.weight_low))))
                output_high = F.relu(torch.spmm(
                    adj_high, (torch.mm(input, self.weight_high))))
                output_mlp = F.relu(torch.mm(input, self.weight_mlp))

            self.att_low, self.att_high, self.att_mlp = self.attention(
                (output_low), (output_high), (output_mlp))
            # 其中att_low、att_high、att_mlp是一个[node_num,1]的tensor,将他们三个合成一个[node_num,3]的score_list

            # gating_scores_list = torch.cat([self.att_low, self.att_high, self.att_mlp], dim=1)
            #
            # self.plot_weight_distribute(data,gating_scores_list)


            return 3*(self.att_low*output_low + self.att_high*output_high + self.att_mlp*output_mlp)
        elif self.model_type == 'acmsgc':
            output_low = torch.spmm(adj_low, torch.mm(input, self.weight_low))
            # torch.mm(input, self.weight_high) - torch.spmm(self.A_EXP,  torch.mm(input, self.weight_high))
            output_high = torch.spmm(
                adj_high,  torch.mm(input, self.weight_high))
            output_mlp = torch.mm(input, self.weight_mlp)

            # self.attention(F.relu(output_low), F.relu(output_high), F.relu(output_mlp))
            self.att_low, self.att_high, self.att_mlp = self.attention(
                (output_low), (output_high), (output_mlp))
            # 3*(output_low + output_high + output_mlp) #
            return 3*(self.att_low*output_low + self.att_high*output_high + self.att_mlp*output_mlp)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


def row_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    adj_normalized = sk_normalize(adj, norm='l1', axis=1)
    return sp.coo_matrix(adj_normalized)

def get_adj_high(adj_low):
    adj_high = -adj_low + sp.eye(adj_low.shape[0])
    return adj_high
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class ACMGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, model_type= 'acmgcn', nlayers=3, variant=False):
        super(ACMGCN, self).__init__()
        self.gcns, self.mlps = nn.ModuleList(), nn.ModuleList()
        self.model_type, self.nlayers, = model_type, nlayers
        if self.model_type == 'mlp':
            self.gcns.append(GraphConvolution(
                in_channels, hidden_channels, model_type=model_type))
            self.gcns.append(GraphConvolution(
                hidden_channels, out_channels, model_type=model_type, output_layer=1))
        elif self.model_type == 'gcn' or self.model_type == 'acmgcn':
            self.gcns.append(GraphConvolution(
                in_channels, hidden_channels,  model_type=model_type, variant=variant))
            self.gcns.append(GraphConvolution(
                hidden_channels, out_channels,  model_type=model_type, output_layer=1, variant=variant))
        elif self.model_type == 'sgc' or self.model_type == 'acmsgc':
            self.gcns.append(GraphConvolution(
                in_channels, out_channels, model_type=model_type))
        elif self.model_type == 'acmsnowball':
            for k in range(nlayers):
                self.gcns.append(GraphConvolution(
                    k * hidden_channels + in_channels, hidden_channels, model_type=model_type, variant=variant))
            self.gcns.append(GraphConvolution(
                nlayers * hidden_channels + in_channels, out_channels, model_type=model_type, variant=variant))
        self.dropout = dropout

    def reset_parameters(self):
        for gcn in self.gcns:
            gcn.reset_parameters()

    def forward(self, data ):
        x = data.graph['node_feat']
        dev = x.device
        edge_index = data.graph['edge_index']
        adj_low = to_scipy_sparse_matrix(edge_index)
        adj_low = row_normalized_adjacency(adj_low)
        adj_high = get_adj_high(adj_low)
        adj_low = sparse_mx_to_torch_sparse_tensor(adj_low).to(dev)
        adj_high = sparse_mx_to_torch_sparse_tensor(adj_high).to(dev)

        if self.model_type == 'acmgcn' or self.model_type == 'acmsgc' or self.model_type == 'acmsnowball':
            x = F.dropout(x, self.dropout, training=self.training)

        if self.model_type == 'acmsnowball':
            list_output_blocks = []
            for layer, layer_num in zip(self.gcns, np.arange(self.nlayers)):
                if layer_num == 0:
                    list_output_blocks.append(F.dropout(
                        F.relu(layer(x, adj_low, adj_high)), self.dropout, training=self.training))
                else:
                    list_output_blocks.append(F.dropout(F.relu(layer(torch.cat(
                        [x] + list_output_blocks[0: layer_num], 1), adj_low, adj_high)), self.dropout, training=self.training))

            return self.gcns[-1](torch.cat([x] + list_output_blocks, 1), adj_low, adj_high)

        fea = (self.gcns[0](x, adj_low, adj_high , data))

        if self.model_type == 'gcn' or self.model_type == 'mlp' or self.model_type == 'acmgcn':
            fea = F.dropout(F.relu(fea), self.dropout, training=self.training)
            fea = self.gcns[-1](fea, adj_low, adj_high , data)
        return fea

class GraphConvolution_HIGH(nn.Module):
    def __init__(self, in_features, out_features,  output_layer=0, variant=False):
        super(GraphConvolution_HIGH, self).__init__()
        self.in_features, self.out_features, self.output_layer,  self.variant = in_features, out_features, output_layer, variant
        self.weight_low, self.weight_high, self.weight_mlp = Parameter(
            torch.FloatTensor(in_features, out_features).cuda()), Parameter(
            torch.FloatTensor(in_features, out_features).cuda()), Parameter(
            torch.FloatTensor(in_features, out_features).cuda())
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_mlp.size(1))
        self.weight_high.data.uniform_(-stdv, stdv)





    def forward(self, input, adj_high):
        if self.variant:
            output_high = (torch.spmm(adj_high, F.relu(torch.mm(input, self.weight_high))))
        else:
            output_high = F.relu(torch.spmm(adj_high, (torch.mm(input, self.weight_high))))
        return output_high

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

class ACMGCN_HIGH(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, nlayers=1, variant=False):
        super(ACMGCN_HIGH, self).__init__()
        self.gcns, self.mlps = nn.ModuleList(), nn.ModuleList()
        self.nlayers = nlayers
        self.gcns.append(GraphConvolution_HIGH(
            in_channels, hidden_channels, variant=variant))
        self.gcns.append(GraphConvolution_HIGH(
            hidden_channels, out_channels, output_layer=1, variant=variant))

        self.dropout = dropout

    def reset_parameters(self):
        for gcn in self.gcns:
            gcn.reset_parameters()

    def forward(self, data):
        x = data.graph['node_feat']
        dev = x.device
        edge_index = data.graph['edge_index']
        adj_low = to_scipy_sparse_matrix(edge_index)
        adj_low = row_normalized_adjacency(adj_low)
        adj_high = get_adj_high(adj_low)
        adj_low = sparse_mx_to_torch_sparse_tensor(adj_low).to(dev)
        adj_high = sparse_mx_to_torch_sparse_tensor(adj_high).to(dev)
        x = F.dropout(x, self.dropout, training=self.training)
        fea = (self.gcns[0](x, adj_high))
        fea = F.dropout(F.relu(fea), self.dropout, training=self.training)
        fea = self.gcns[-1](fea, adj_high)

        return fea

class GraphConvolution_LOW(nn.Module):
    def __init__(self, in_features, out_features,  output_layer=0, variant=False):
        super(GraphConvolution_LOW, self).__init__()
        self.in_features, self.out_features, self.output_layer,  self.variant = in_features, out_features, output_layer, variant
        self.weight_low, self.weight_high, self.weight_mlp = Parameter(
            torch.FloatTensor(in_features, out_features).cuda()), Parameter(
            torch.FloatTensor(in_features, out_features).cuda()), Parameter(
            torch.FloatTensor(in_features, out_features).cuda())
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_mlp.size(1))
        self.weight_low.data.uniform_(-stdv, stdv)


    def forward(self, input, adj_low):
        output_low = torch.mm(adj_low, torch.mm(input, self.weight_low))
        return output_low

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

class ACMGCN_LOW(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, nlayers=1, variant=False):
        super(ACMGCN_LOW, self).__init__()
        self.gcns, self.mlps = nn.ModuleList(), nn.ModuleList()
        self.nlayers = nlayers
        self.gcns.append(GraphConvolution_LOW(
            in_channels, hidden_channels, variant=variant))
        self.gcns.append(GraphConvolution_LOW(
            hidden_channels, out_channels, output_layer=1, variant=variant))

        self.dropout = dropout

    def reset_parameters(self):
        for gcn in self.gcns:
            gcn.reset_parameters()

    def forward(self, data):
        x = data.graph['node_feat']
        dev = x.device
        edge_index = data.graph['edge_index']
        adj_low = to_scipy_sparse_matrix(edge_index)
        adj_low = row_normalized_adjacency(adj_low)
        adj_high = get_adj_high(adj_low)
        adj_low = sparse_mx_to_torch_sparse_tensor(adj_low).to(dev)
        adj_high = sparse_mx_to_torch_sparse_tensor(adj_high).to(dev)

        x = F.dropout(x, self.dropout, training=self.training)


        fea = (self.gcns[0](x, adj_low))
        fea = F.dropout(F.relu(fea), self.dropout, training=self.training)
        fea = self.gcns[-1](fea, adj_low)

        return fea




class MLPNORM(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, nnodes,  dropout, alpha = 1.0, beta = 1.0, gamma = 0, delta = 0,
                 norm_func_id = 1, norm_layers = 2, orders = 1, orders_func_id = 2):
        super(MLPNORM, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.fc3 = nn.Linear(nnodes, hidden_channels)
        self.out_channels = out_channels
        self.dropout = dropout
        self.alpha = torch.tensor(alpha).to(device)
        self.beta = torch.tensor(beta).to(device)
        self.gamma = torch.tensor(gamma).to(device)
        self.delta = torch.tensor(delta).to(device)
        self.norm_layers = norm_layers
        self.orders = orders
        self.device = device
        self.class_eye = torch.eye(self.out_channels).to(device)
        self.nodes_eye = torch.eye(nnodes).to(device)
        self.orders_weight = Parameter(
            (torch.ones(orders, 1) / orders).to(device), requires_grad=True
        )
        self.orders_weight_matrix = Parameter(
            torch.DoubleTensor(out_channels, orders).to(device), requires_grad=True
        )
        self.orders_weight_matrix2 = Parameter(
            torch.DoubleTensor(orders, orders).to(device), requires_grad=True
        )
        self.diag_weight = Parameter(
            (torch.ones(out_channels, 1) / out_channels).to(device), requires_grad=True
        )
        init.kaiming_normal_(self.orders_weight_matrix, mode='fan_out')
        init.kaiming_normal_(self.orders_weight_matrix2, mode='fan_out')
        self.elu = torch.nn.ELU()
        if norm_func_id == 1:
            self.norm = self.norm_func1
        else:
            self.norm = self.norm_func2

        if orders_func_id == 1:
            self.order_func = self.order_func1
        elif orders_func_id == 2:
            self.order_func = self.order_func2
        else:
            self.order_func = self.order_func3

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.orders_weight = Parameter(
            (torch.ones(self.orders, 1) / self.orders).to(self.device), requires_grad=True
        )
        init.kaiming_normal_(self.orders_weight_matrix, mode='fan_out')
        init.kaiming_normal_(self.orders_weight_matrix2, mode='fan_out')
        self.diag_weight = Parameter(
            (torch.ones(self.out_channels, 1) / self.out_channels).to(self.device), requires_grad=True
        )

    def forward(self, data):
        x = data.graph['node_feat']
        edge_index = data.graph['edge_index']
        adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(
            data.graph['num_nodes'], data.graph['num_nodes'])).to_torch_sparse_coo_tensor()
        adj = adj.to(device)


        xX = self.fc1(x)
        xA = self.fc3(adj.float())
        x = F.relu(self.delta * xX + (1-self.delta) * xA)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        h0 = x
        for _ in range(self.norm_layers):
            x = self.norm(x, h0, adj)
        return x

    def norm_func1(self, x, h0, adj):
        coe = 1.0 / (self.alpha + self.beta)
        coe1 = 1.0 - self.gamma
        coe2 = 1.0 / coe1
        res = torch.mm(torch.transpose(x, 0, 1), x)
        inv = torch.inverse(coe2 * coe2 * self.class_eye + coe * res)
        res = torch.mm(inv, res)
        res = coe1 * coe * x - coe1 * coe * coe * torch.mm(x, res)
        tmp = torch.mm(torch.transpose(x, 0, 1), res)
        sum_orders = self.order_func(x, res, adj)
        res = coe1 * torch.mm(x, tmp) + self.beta * sum_orders - \
            self.gamma * coe1 * torch.mm(h0, tmp) + self.gamma * h0
        return res

    def norm_func2(self, x, h0, adj):
        # print('norm_func2 run')
        coe = 1.0 / (self.alpha + self.beta)
        coe1 = 1 - self.gamma
        coe2 = 1.0 / coe1
        res = torch.mm(torch.transpose(x, 0, 1), x)
        inv = torch.inverse(coe2 * coe2 * self.class_eye + coe * res)
        # u = torch.cholesky(coe2 * coe2 * torch.eye(self.nclass) + coe * res)
        # inv = torch.cholesky_inverse(u)
        res = torch.mm(inv, res)
        res = (coe1 * coe * x -
               coe1 * coe * coe * torch.mm(x, res)) * self.diag_weight.t()
        tmp = self.diag_weight * (torch.mm(torch.transpose(x, 0, 1), res))
        sum_orders = self.order_func(x, res, adj)
        res = coe1 * torch.mm(x, tmp) + self.beta * sum_orders - \
              self.gamma * coe1 * torch.mm(h0, tmp) + self.gamma * h0

        # calculate z
        xx = torch.mm(x, x.t())
        hx = torch.mm(h0, x.t())
        # print('adj', adj.shape)
        # print('orders_weight', self.orders_weight[0].shape)
        adj = adj.to_dense()
        adjk = adj
        a_sum = adjk * self.orders_weight[0]
        for i in range(1, self.orders):
            adjk = torch.mm(adjk, adj)
            a_sum += adjk * self.orders_weight[i]
        z = torch.mm(coe1 * xx + self.beta * a_sum - self.gamma * coe1 * hx,
                     torch.inverse(coe1 * coe1 * xx + (self.alpha + self.beta) * self.nodes_eye))
        # print(z.shape)
        # print(z)
        return res

    def order_func1(self, x, res, adj):
        tmp_orders = res
        sum_orders = tmp_orders
        for _ in range(self.orders):
            # tmp_orders = torch.sparse.spmm(adj, tmp_orders)
            tmp_orders = adj.matmul(tmp_orders)
            sum_orders = sum_orders + tmp_orders
        return sum_orders

    def order_func2(self, x, res, adj):
        # tmp_orders = torch.sparse.spmm(adj, res)
        tmp_orders = adj.matmul(res)
        sum_orders = tmp_orders * self.orders_weight[0]
        for i in range(1, self.orders):
            # tmp_orders = torch.sparse.spmm(adj, tmp_orders)
            tmp_orders = adj.matmul(tmp_orders)
            sum_orders = sum_orders + tmp_orders * self.orders_weight[i]
        return sum_orders

    def order_func3(self, x, res, adj):
        orders_para = torch.mm(torch.relu(torch.mm(x, self.orders_weight_matrix)),
                               self.orders_weight_matrix2)
        orders_para = torch.transpose(orders_para, 0, 1)
        # tmp_orders = torch.sparse.spmm(adj, res)
        tmp_orders = adj.matmul(res)
        sum_orders = orders_para[0].unsqueeze(1) * tmp_orders
        for i in range(1, self.orders):
            # tmp_orders = torch.sparse.spmm(adj, tmp_orders)
            tmp_orders = adj.matmul(tmp_orders)
            sum_orders = sum_orders + orders_para[i].unsqueeze(1) * tmp_orders
        return sum_orders




