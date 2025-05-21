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
import time

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
    args = OmegaConf.merge( conf_yaml,conf_cli)

    # setup_seed(args.seed)
    # TODO 检测一下GPU和CPU的使用率观测能否跑满
    #单卡训练
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    ### Load and preprocess data ###
    dataset = load_nc_dataset(args.dataset)


    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    #获取OOD数据集
    train_idx, val_idx, test_idx = ood_split(dataset, args.domain, args.shift)
    split_idx_lst = [{'train': train_idx, 'valid': val_idx, 'test': test_idx}]

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

    for run in range(args.runs):
        setup_seed(run)
        model = parse_method(args, dataset, n, c, d,run, device)
        split_idx = split_idx_lst[0]
        train_idx = torch.tensor(split_idx['train']).to(device)
        test_idx = torch.tensor(split_idx['test']).to(device)
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

        path = f'./models/{args.dataset}/{run}/{args.method}_{args.domain}_{args.shift}_{args.trail}.pt'
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        if args.debug:
            writer = SummaryWriter(log_dir="tensorboard/joint_training")

        # 检测之前有没有训练过，保存的最好模型可以直接加载
        if os.path.exists(path):
            print(f"Loading model from {path}")
            model = torch.load(path, weights_only=False)
            result = evaluate(model, dataset, split_idx, eval_func)
            print(f'Train Acc: {100 * result[0]:.2f}%, '
                  f'Valid Acc: {100 * result[1]:.2f}%, '
                  f'Test Acc: {100 * result[2]:.2f}%')
        else:
            print(f"Training model from scratch and saving to {path}")
            for epoch in range(args.epochs):
                model.train()
                optimizer.zero_grad()
                out = model(dataset)
                out = F.log_softmax(out, dim=1)
                # main_loss
                label_loss = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx])
                # align_loss
                node_patterns = model.gating_network.get_embed(feat, edge_index)
                expert_weights = model.gating_network(dataset)
                align_loss = compute_soft_kmeans_align_loss(node_patterns, expert_weights,
                                                            num_clusters=args.num_clusters,
                                                            alpha=args.alpha,
                                                            max_iter=args.max_iter,
                                                            use_match_matrix=args.use_match_matrix,
                                                            device=device,
                                                            plot=False,
                                                            idx=train_idx)
                # 合并总损失（可根据需要加权各部分）
                total_loss = label_loss + args.align_loss_weight * align_loss
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
                if args.debug:
                    # 记录到 TensorBoard
                    writer.add_scalar("Loss/Supervised", label_loss.item(), epoch)
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
            logger.print_statistics(run)
            if args.TTT == False:
                test_list.append(100 * best_test_performance)
                test_list = np.array(test_list)
                print(f'All runs:')
                print(f'Final Test: {test_list.mean():.2f} ± {test_list.std():.2f}')

        if args.debug:
            writer.close()


        if args.TTT == True:
            model = torch.load(path, weights_only=False)
            ### TEST TIME TRAINING ###
            print("====test time training====")
            if args.debug:
                writer = SummaryWriter(log_dir="tensorboard/test_time_training")


            #TODO 可以调整ttt去调哪些参数
            params = list(model.gating_network.parameters())
            optimizer = torch.optim.Adam(params, lr=args.ttt_lr,  weight_decay=args.ttt_weight_decay)

            for i in range(args.ttt_epochs):
                model.train()
                optimizer.zero_grad()
                node_patterns = model.gating_network.get_embed(feat, edge_index)
                expert_weights = model.gating_network(dataset)

                ssl_loss = compute_soft_kmeans_align_loss(node_patterns, expert_weights, num_clusters=args.ttt_num_clusters, alpha=args.ttt_alpha, max_iter=args.ttt_max_iter, use_match_matrix=args.ttt_use_match_matrix, device=device,plot= False,idx=test_idx)
                ssl_loss.backward()
                #可以使用梯度裁剪来防止梯度溢出,对于 L2 范数的裁剪
                nn.utils.clip_grad_norm_(model.gating_network.parameters(), max_norm=1.0)
                optimizer.step()
                if args.debug:
                    # 记录 SSL Loss 到 TensorBoard
                    writer.add_scalar("Loss/SSL", ssl_loss.item(), i)
                    print(f'Epoch {i}: {ssl_loss.item()}')
                    # 记录 Gating Network 梯度范数
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            writer.add_scalar(f"Gradient/TTT/{name}", param.grad.norm().item(), i)

                result = evaluate(model, dataset, split_idx, eval_func)

                # 根据val,test save
                val_performance = result[1]
                test_performance = result[2]
                if val_performance >= best_val_performance:  # 如果当前epoch的validation性能更好
                    if test_performance >= best_test_performance:
                        best_val_performance = val_performance  # 更新最佳validation性能
                        best_test_performance = test_performance

                if args.debug:
                    writer.add_scalar("Accuracy/Test", 100 * result[2], i)

                    print(f'Epoch: {i}, '
                          f'SSL Loss: {ssl_loss:.4f}, '
                          f'Test: {100 * result[2]:.2f}%')

            if args.debug:
                writer.close()

            test_list.append(100 * best_test_performance)
            test_list = np.array(test_list)
            print(f'Final TTT Test: {test_list.mean():.2f} ± {test_list.std():.2f}')

if __name__ == "__main__":
    main()

