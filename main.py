## AI框架库
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch.utils.tensorboard import SummaryWriter

## 第三方辅助库
import numpy as np
import random
import os
from omegaconf import OmegaConf
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore",category=UserWarning)


## 项目通过辅助库
from parse import parse_method
from logger import Logger
#数据集相关库
from data_utils import load_fixed_splits, eval_acc,  evaluate, get_homophily_split
from dataset import load_nc_dataset
from ttt import compute_soft_kmeans_align_loss
from ood_split import  ood_split
from homo_utils import compute_homo

# 固定参数，保证实验的可复现性
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False



def main():
    ## 参数配置与管理
    conf_cli = OmegaConf.from_cli()
    conf_yaml = OmegaConf.load(conf_cli.config_path)
    args = OmegaConf.merge(conf_cli, conf_yaml)

    setup_seed(args.seed)
    #单卡训练
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    ### Load and preprocess data ###
    dataset = load_nc_dataset(args.dataset)


    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    #structure Shift
    if args.rand_split:
        split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop) for _ in range(args.runs)]
    elif args.ood_homophily_split:
        train_idx, val_idx, test_idx = ood_split(dataset, 'degree', 'concept')
        split_idx_lst = [{'train': train_idx, 'valid': val_idx, 'test': test_idx}]
    else:
        split_idx_lst = [load_fixed_splits(args.dataset, i) for i in range(args.runs)]
 




    n = dataset.graph['num_nodes']
    # 标签可能有独热编码和非独热编码两种形式
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
    dataset.graph['edge_index'], dataset.graph['node_feat'] = dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)
    dataset.label = dataset.label.to(device)
    feat = dataset.graph['node_feat']
    edge_index = dataset.graph['edge_index']

    ### Load method ###
    criterion = nn.NLLLoss()
    eval_func = eval_acc
    logger = Logger(args.runs, args)
    test_list = []
    ### Training loop ###
    for run in range(args.runs):
        model = parse_method(args, dataset, n, c, d,run, device)
        split_idx = split_idx_lst[run]
        train_idx = torch.tensor(split_idx['train']).to(device)
        test_idx = torch.tensor(split_idx['test']).to(device)

        # 从train_idx中随机选择50%的节点作为新的train_idx
        # num_train = len(train_idx)
        # random_indices = torch.randperm(num_train)
        # half_size = num_train // 2
        # train_idx = train_idx[random_indices[:half_size]]

        #重置参数
        model.reset_parameters()

        #设置optimizer
        if args.method == 'MoEGCN':
            # 将模型的 expert 和 gating 参数分开
            expert_params = list(model.expert1.parameters()) + list(model.expert2.parameters()) + list(model.expert3.parameters())+ list(model.expert4.parameters()) + list(model.expert5.parameters())
            gating_params = list(model.gating_network.parameters())

            # 定义不同的超参数
            expert_lr = args.lr
            expert_weight_decay = args.weight_decay
            gating_lr = args.gating_lr
            gating_weight_decay = args.gating_weight_decay

            # 创建优化器
            optimizer = torch.optim.AdamW([
                {'params': expert_params, 'lr': expert_lr, 'weight_decay': expert_weight_decay},
                {'params': gating_params, 'lr': gating_lr, 'weight_decay': gating_weight_decay},
            ])
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_val_performance = 0  # 初始化最佳validation性能
        best_test_performance = 0

        path = f'./models/{args.dataset}/{run}/{args.method}_{args.trail}.pt'
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        # 初始化 TensorBoard 写入器
        writer = SummaryWriter(log_dir="tensorboard/joint_training")

        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()

            out = model(dataset)
            out = F.log_softmax(out, dim=1)
            # main_loss
            label_loss = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx])

            # align_loss
            node_patterns = model.gating_network.get_embed(feat,edge_index)
            expert_weights = model.gating_network(dataset)
            align_loss, c_np, c_ew , M_params = compute_soft_kmeans_align_loss(node_patterns, expert_weights, num_clusters=args.num_clusters, alpha=10, max_iter=5, use_match_matrix=True, device=device, detach_centers=True,plot=False)


            # 合并总损失（可根据需要加权各部分）

            total_loss = label_loss +args.align_loss_weight * align_loss


            total_loss.backward()
            optimizer.step()

            result = evaluate(model, dataset, split_idx, eval_func)
            logger.add_result(run, result[:-1])

            # 根据val,test save
            val_performance = result[1]
            test_performance = result[2]
            if val_performance >= best_val_performance:  # 如果当前epoch的validation性能更好
                if test_performance >= best_test_performance:
                    best_val_performance = val_performance  # 更新最佳validation性能
                    best_test_performance = test_performance
                    torch.save(model, path)

            # 记录到 TensorBoard
            writer.add_scalar("Loss/Supervised", label_loss.item(), epoch)
            # writer.add_scalar("Loss/SSL", args.tent_loss_weight, epoch)
            writer.add_scalar("Loss/Total", total_loss.item(), epoch)
            writer.add_scalar("Accuracy/Train", 100 * result[0], epoch)
            writer.add_scalar("Accuracy/Valid", 100 * result[1], epoch)
            writer.add_scalar("Accuracy/Test", 100 * result[2], epoch)


            # 记录梯度
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_scalar(f"Gradient/MoETTT/{name}", param.grad.norm().item(), epoch)

            # 打印训练信息
            if epoch % args.display_step == 0:
                print(f'Epoch: {epoch:02d}, '
                      f'Total Loss: {total_loss:.4f}, '
                      f'Train Acc: {100 * result[0]:.2f}%, '
                      f'Valid Acc: {100 * result[1]:.2f}%, '
                      f'Test Acc: {100 * result[2]:.2f}%')

        writer.close()
        test_list.append(100 * best_test_performance)
        logger.print_statistics(run)
        test_list = np.array(test_list)
        print(f'All runs:')
        print(f'Final Test: {test_list.mean():.2f} ± {test_list.std():.2f}')

        if args.TTT == True:
        
            ### TEST TIME TRAINING ###
            print("====test time training====")
            writer = SummaryWriter(log_dir="tensorboard/test_time_training")
            model.train()
            params = list(model.gating_network.parameters()) +[model.routing_centers]
            optimizer = torch.optim.Adam(params, lr=0.0005, betas=(0.9, 0.999), weight_decay=0)
            for i in range(33):
                optimizer.zero_grad()
                out = model(dataset)
                out = F.log_softmax(out, dim=1)

                ssl_loss =None

    
                # 记录 SSL Loss 到 TensorBoard
                writer.add_scalar("Loss/SSL", ssl_loss.item(), i)
    
                print(f'Epoch {i}: {ssl_loss.item()}')
                ssl_loss.backward()
    
                # 记录 Gating Network 梯度范数
                for name, param in model.gating_network.named_parameters():
                    if param.grad is not None:
                        writer.add_scalar(f"Gradient/GatingNetwork/{name}", param.grad.norm().item(), i)
    
    
                if model.routing_centers.grad is not None:
                    writer.add_scalar("Gradient/routing_centers", model.routing_centers.grad.norm().item(), i)
    
                optimizer.step()
                result = evaluate(model, dataset, split_idx, eval_func)
                writer.add_scalar("Accuracy/Train", 100 * result[0], i)
                writer.add_scalar("Accuracy/Valid", 100 * result[1], i)
                writer.add_scalar("Accuracy/Test", 100 * result[2], i)
    
                print(f'Test Time Training0Epoch: {i:02d}, '
                      f'SSL Loss: {ssl_loss:.4f}, '
                      f'Train: {100 * result[0]:.2f}%, '
                      f'Valid: {100 * result[1]:.2f}%, '
                      f'Test: {100 * result[2]:.2f}%')
    
            writer.close()
    
            test_list.append(100 * best_test_performance)
            logger.print_statistics(run)
            test_list = np.array(test_list)
            print(f'All runs:')
            print(f'Final Test: {test_list.mean():.2f} ± {test_list.std():.2f}')



if __name__ == "__main__":
    main()

