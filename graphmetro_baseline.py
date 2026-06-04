import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_adj


class SimpleGCN(nn.Module):
    """GCN that operates on raw tensors (x, edge_index) for use as expert."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SimpleGCN, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GraphMETROGating(nn.Module):
    """Gating network: encodes transformed graph to predict expert routing."""
    def __init__(self, in_channels, hidden_channels, num_experts, dropout):
        super(GraphMETROGating, self).__init__()
        self.encoder = SimpleGCN(in_channels, hidden_channels, hidden_channels, num_layers=2, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_experts),
        )

    def reset_parameters(self):
        self.encoder.reset_parameters()
        for layer in self.head:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x, edge_index):
        emb = self.encoder(x, edge_index)
        logits = self.head(emb)
        return logits


class GraphMETROModel(nn.Module):
    """GraphMETRO: MoE with gating trained via graph transformations.

    During training, a random graph transformation is applied. The gating
    network learns to predict which transform was applied, and routes to
    the appropriate expert. During inference, expert weights are predicted
    directly from the clean graph.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_experts,
                 num_layers, dropout):
        super(GraphMETROModel, self).__init__()
        self.num_experts = num_experts
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.experts = nn.ModuleList([
            SimpleGCN(in_channels, hidden_channels, hidden_channels, num_layers, dropout)
            for _ in range(num_experts)
        ])
        self.gating = GraphMETROGating(in_channels, hidden_channels, num_experts, dropout)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for expert in self.experts:
            expert.reset_parameters()
        self.gating.reset_parameters()
        self.classifier.reset_parameters()

    @staticmethod
    def apply_transform(x, edge_index, transform_idx, drop_p=0.1, mask_p=0.1, noise_std=0.1):
        """Apply one of K graph transformations.

        transform_idx:
            0 - identity (no transform)
            1 - drop_edge: randomly remove edges
            2 - mask_node_feat: randomly zero out node features
            3 - noisy_node_feat: add Gaussian noise to node features
        """
        if transform_idx == 0:
            return x, edge_index
        elif transform_idx == 1:
            edge_index_t, _ = dropout_adj(edge_index, p=drop_p, training=True)
            return x, edge_index_t
        elif transform_idx == 2:
            mask = torch.rand_like(x) > mask_p
            return x * mask, edge_index
        elif transform_idx == 3:
            noise = torch.randn_like(x) * noise_std
            return x + noise, edge_index
        else:
            return x, edge_index

    def forward(self, data, transform_idx=None, return_gating=False):
        x = data.graph['node_feat']
        edge_index = data.graph['edge_index']

        if self.training and transform_idx is not None:
            x_t, edge_index_t = self.apply_transform(x, edge_index, transform_idx)
        else:
            x_t, edge_index_t = x, edge_index

        gate_logits = self.gating(x_t, edge_index_t)
        gate_weights = F.softmax(gate_logits, dim=-1)

        expert_outs = []
        for expert in self.experts:
            out = expert(x_t, edge_index_t)
            expert_outs.append(out)
        expert_outs = torch.stack(expert_outs, dim=0)

        gate_weights_t = gate_weights.t().unsqueeze(-1)
        combined = torch.sum(expert_outs * gate_weights_t, dim=0)

        out = self.classifier(combined)
        if return_gating:
            return out, gate_logits
        return out
