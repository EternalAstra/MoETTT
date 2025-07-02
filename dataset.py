import torch
from load_data import DATAPATH
from data_utils import rand_train_test_idx
from torch_geometric.datasets import CitationFull,Planetoid,WikiCS
from data_utils import load_data_new
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T

class NCDataset(object):
    def __init__(self, name, root=f'{DATAPATH}'):

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.6, valid_prop=.2):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_nc_dataset(dataname):
    """ Loader for NCDataset, returns NCDataset. """
    if dataname in ('Cora', 'CiteSeer', 'PubMed'):
        dataset = load_planetoid_dataset(dataname)
    elif dataname in ('chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin'):
        dataset = load_geom_gcn_dataset(dataname)
    elif dataname == 'ogbn-arxiv':
        dataset = load_arxiv_dataset('ogbn-arxiv')
    elif dataname == 'wikics':
        dataset = load_wikics_dataset('wikics')
    return dataset


def load_planetoid_dataset(name):
    torch_dataset = Planetoid(root=f'{DATAPATH}/Planetoid',
                              name=name)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes
    # print(f"Num nodes: {num_nodes}")

    dataset = NCDataset(name)

    dataset.train_idx = torch.where(data.train_mask)[0]
    dataset.valid_idx = torch.where(data.val_mask)[0]
    dataset.test_idx = torch.where(data.test_mask)[0]

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}

    def planetoid_orig_split(**kwargs):
        return {'train': torch.as_tensor(dataset.train_idx),
                'valid': torch.as_tensor(dataset.valid_idx),
                'test': torch.as_tensor(dataset.test_idx)}

    dataset.get_idx_split = planetoid_orig_split
    dataset.label = label

    return dataset


def load_wikics_dataset(name):
    #倒入WikiCS数据集
    torch_dataset = WikiCS(root=f'{DATAPATH}/WikiCS')
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes
    dataset = NCDataset(name)
    dataset.train_idx = torch.where(data.train_mask)[0]
    dataset.valid_idx = torch.where(data.val_mask)[0]
    dataset.test_idx = torch.where(data.test_mask)[0]

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}

    dataset.label = label
    return dataset


def load_arxiv_dataset(name):
    torch_dataset = PygNodePropPredDataset(root=f'{DATAPATH}/arxiv', name='ogbn-arxiv')
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes
    dataset = NCDataset(name)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}

    dataset.label = label
    return dataset


def load_geom_gcn_dataset(name):
    # Load data
    edge_index, node_feat, label, _, _, _ = load_data_new(name)
    num_nodes = node_feat.shape[0]

    dataset = NCDataset(name)

    node_feat = torch.tensor(node_feat, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.int64)
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    label = torch.tensor(label, dtype=torch.long)
    dataset.label = label
    return dataset



