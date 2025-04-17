import json
import itertools
import random
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset,  Data
from torch_geometric.utils import degree, homophily


#获取homophily
def homophily(edge_index, y):
    num_nodes = len(y)
    homophily_counter = torch.zeros(num_nodes,dtype=torch.float)
    edge_counter = torch.zeros(num_nodes,dtype=torch.float)

    for i in  range(edge_index.size(1)):

        source = edge_index[0,i].item()
        target = edge_index[1,i].item()

        edge_counter[source] += 1
        edge_counter[target] += 1

        if y[source] == y[target]:
            homophily_counter[source] += 1
            homophily_counter[target] += 1

    homophily_values =  homophily_counter/edge_counter

    homophily_values[torch.isnan(homophily_values)] = 0

    return homophily_values


class DomainGetter(object):
    def __init__(self):
        pass

    def get_homophily(self,graph: Data) -> int:
        try:
            node_homophily = homophily(graph.graph['edge_index'] ,graph.label)
            return node_homophily
        except ValueError as e:
            print('#E#Get homophily error.')
            raise e

    def get_degree(self, graph: Data) -> int:
        try:
            node_degree = degree(graph.graph['edge_index'][0], graph.graph['num_nodes'])
            return node_degree
        except ValueError as e:
            print('#E#Get degree error.')
            raise e

    def get_word(self, graph: Data) -> int:
        num_word = graph.graph['node_feat'].sum(1)
        return num_word


class DataInfo(object):
    r"""
    The class for data point storage. This enables tackling node data point like graph data point, facilitating data splits.
    """
    def __init__(self, idx, y):
        super(DataInfo, self).__init__()
        self.storage = []
        self.idx = idx
        self.y = y

    def __repr__(self):
        s = [f'{key}={self.__getattribute__(key)}' for key in self.storage]
        s = ', '.join(s)
        return f"DataInfo({s})"

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key != 'storage':
            self.storage.append(key)


def get_domain_sorted_indices(graph, domain, num_data):
    domain_getter = DomainGetter()
    graph.__setattr__(domain, getattr(domain_getter, f'get_{domain}')(graph))

    data_list = []
    for i in range(num_data):
        data_info = DataInfo(idx=i, y=graph.label[i])
        data_info.__setattr__(domain, getattr(graph,domain)[i])
        data_list.append(data_info)

    sorted_data_list = sorted(data_list, key=lambda data: getattr(data, domain))

    # Assign domain id
    cur_domain_id = -1
    cur_domain = None
    sorted_domain_split_data_list = []
    for data in sorted_data_list:
        if getattr(data, domain) != cur_domain:
            cur_domain = getattr(data, domain)
            cur_domain_id += 1
            sorted_domain_split_data_list.append([])
        data.domain_id = torch.LongTensor([cur_domain_id])
        sorted_domain_split_data_list[data.domain_id].append(data)

    return sorted_data_list, sorted_domain_split_data_list


def ood_split(data ,domain, shift):
    num_data = data.graph['node_feat'].shape[0]
    sorted_data_list, sorted_domain_split_data_list = get_domain_sorted_indices(data, domain,num_data)
    if shift == 'covariate':
        if domain == 'degree':
            sorted_data_list = sorted_data_list[::-1]
            train_ratio = 0.6
            val_ratio = 0.2
            id_test_ratio = 0.1
        else:
            sorted_data_list = sorted_data_list[::-1]
            train_ratio = 0.6
            val_ratio = 0.2
            id_test_ratio = 0.1

        train_split = int(num_data * train_ratio)
        val_split = int(num_data * (train_ratio + val_ratio))

        train_val_test_split = [0, train_split, val_split]
        train_val_test_list = [[], [], []]
        cur_env_id = -1
        cur_domain_id = None
        for i, data in enumerate(sorted_data_list):
            if cur_env_id < 2 and i >= train_val_test_split[cur_env_id + 1] and data.domain_id != cur_domain_id:
                # if i >= (cur_env_id + 1) * num_per_env:
                cur_env_id += 1
            cur_domain_id = data.domain_id
            train_val_test_list[cur_env_id].append(data)

        train_list, ood_val_list, ood_test_list = train_val_test_list
        # 提取 idx 数组并转换
        train_ids = [data_info.idx for data_info in train_list]
        ood_val_ids = [data_info.idx for data_info in ood_val_list]
        ood_test_ids = [data_info.idx for data_info in ood_test_list]

        # 转换为 ndarray 数组
        train_ids_array = np.array(train_ids)
        ood_val_ids_array = np.array(ood_val_ids)
        ood_test_ids_array = np.array(ood_test_ids)

        return train_ids_array, ood_val_ids_array, ood_test_ids_array

    else:
        global_pyx = []
        for each_domain_datas in tqdm(sorted_domain_split_data_list):
            pyx = []
            for data in each_domain_datas:
                data.pyx = torch.tensor(np.nanmean(data.y).item())
                if torch.isnan(data.pyx):
                    data.pyx = torch.tensor(0.)
                pyx.append(data.pyx.item())
                global_pyx.append(data.pyx.item())
            pyx = sum(pyx) / each_domain_datas.__len__()
            each_domain_datas.append(pyx)

        global_mean_pyx = np.mean(global_pyx)
        global_mid_pyx = np.sort(global_pyx)[len(global_pyx) // 2]


        bias_connect = [0.95, 0.95, 0.9, 0.85, 0.5]
        is_train_split = [True, False, True, True, False]
        is_val_split = [False if i < len(is_train_split) - 1 else True for i in range(len(is_train_split))]
        is_test_split = [not (tr_sp or val_sp) for tr_sp, val_sp in zip(is_train_split, is_val_split)]

        split_picking_ratio = [0.4, 0.6, 0.5, 1, 1]

        order_connect = [[] for _ in range(len(bias_connect))]
        cur_num = 0
        for i in range(len(sorted_domain_split_data_list)):
            randc = 1 if cur_num < num_data / 2 else - 1
            cur_num += sorted_domain_split_data_list[i].__len__() - 1
            for j in range(len(order_connect)):
                order_connect[j].append(randc if is_train_split[j] else - randc)

        env_list = [[] for _ in range(len(bias_connect))]
        cur_split = 0
        env_id = -1
        while cur_split < len(env_list):
            if is_train_split[cur_split]:
                env_id += 1
            next_split = False

            for domain_id, each_domain_datas in enumerate(sorted_domain_split_data_list):
                pyx_mean = each_domain_datas[-1]
                pop_items = []
                both_label_domain = [False, False]
                label_data_candidate = [None, None]
                both_label_include = [False, False]
                for i in range(len(each_domain_datas) - 1):
                    data = each_domain_datas[i]
                    picking_rand = random.random()
                    data_rand = random.random()  # random num for data point
                    if cur_split == len(env_list) - 1:
                        data.env_id = env_id
                        env_list[cur_split].append(data)
                        pop_items.append(data)
                    else:
                        if order_connect[cur_split][domain_id] * (data.pyx - global_mean_pyx) > 0:
                            both_label_domain[0] = True
                            if data_rand < bias_connect[cur_split] and picking_rand < split_picking_ratio[cur_split]:
                                both_label_include[0] = True
                                data.env_id = env_id
                                env_list[cur_split].append(data)
                                pop_items.append(data)
                            else:
                                label_data_candidate[0] = data
                        else:
                            both_label_domain[1] = True
                            if data_rand > bias_connect[cur_split] and picking_rand < split_picking_ratio[cur_split]:
                                both_label_include[1] = True
                                data.env_id = env_id
                                env_list[cur_split].append(data)
                                pop_items.append(data)
                            else:
                                label_data_candidate[1] = data

                # --- Add extra data: avoid extreme label imbalance ---
                if both_label_domain[0] and both_label_domain[1] and (both_label_include[0] or both_label_include[1]):
                    extra_data = None
                    if not both_label_include[0]:
                        extra_data = label_data_candidate[0]
                    if not both_label_include[1]:
                        extra_data = label_data_candidate[1]
                    if extra_data:
                        extra_data.env_id = env_id
                        env_list[cur_split].append(extra_data)
                        pop_items.append(extra_data)
                for pop_item in pop_items:
                    each_domain_datas.remove(pop_item)

            cur_split += 1
            num_train = sum([len(env) for i, env in enumerate(env_list) if is_train_split[i]])
            num_val = sum([len(env) for i, env in enumerate(env_list) if is_val_split[i]])
            num_test = sum([len(env) for i, env in enumerate(env_list) if is_test_split[i]])
            print("#D#train: %d, val: %d, test: %d" % (num_train, num_val, num_test))

        train_list, ood_val_list, ood_test_list = list(
            itertools.chain(*[env for i, env in enumerate(env_list) if is_train_split[i]])), \
            list(itertools.chain(
                *[env for i, env in enumerate(env_list) if is_val_split[i]])), \
            list(itertools.chain(
                *[env for i, env in enumerate(env_list) if is_test_split[i]]))

        # 提取 idx 数组并转换
        train_ids = [data_info.idx for data_info in train_list]
        ood_val_ids = [data_info.idx for data_info in ood_val_list]
        ood_test_ids = [data_info.idx for data_info in ood_test_list]

        # 转换为 ndarray 数组
        train_ids_array = np.array(train_ids)
        ood_val_ids_array = np.array(ood_val_ids)
        ood_test_ids_array = np.array(ood_test_ids)

        return train_ids_array, ood_val_ids_array, ood_test_ids_array

