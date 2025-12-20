"""
CausalGAT Model - Causal Attention Graph Neural Network for Vulnerability Detection.

This model uses:
- Graph Convolutional Networks (GCN) for structural features
- Graph Attention Networks (GAT) for attention-based aggregation
- Causal intervention to separate context and object features
- Multi-modal fusion of code embeddings and graph embeddings

Reference: https://github.com/yongduosui/CAL/tree/main
"""

import random
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Linear, BatchNorm1d


# Check if torch_geometric is available
try:
    from torch_scatter import scatter_add
    from torch_geometric.nn import MessagePassing, GATConv, global_add_pool
    from torch_geometric.utils import add_self_loops, remove_self_loops
    from torch_geometric.data import Data
    from torch_geometric.nn.inits import glorot, zeros
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    MessagePassing = nn.Module  # Fallback for type hints


class GCNConv(MessagePassing if TORCH_GEOMETRIC_AVAILABLE else nn.Module):
    """
    Graph Convolutional Network layer with optional edge normalization.
    
    Supports GFN (Graph Feature Network) mode where no message passing occurs.
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 improved: bool = False,
                 cached: bool = False,
                 bias: bool = True,
                 edge_norm: bool = True,
                 gfn: bool = False):
        if TORCH_GEOMETRIC_AVAILABLE:
            super(GCNConv, self).__init__('add')
        else:
            super(GCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.edge_norm = edge_norm
        self.gfn = gfn
        self.message_mask = None
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)
        
        edge_weight = edge_weight.view(-1)
        
        assert edge_weight.size(0) == edge_index.size(1)
        
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        
        # Add edge_weight for loop edges.
        loop_weight = torch.full((num_nodes, ),
                                 1 if not improved else 2,
                                 dtype=edge_weight.dtype,
                                 device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        x = torch.matmul(x, self.weight)
        if self.gfn:
            return x
    
        if not self.cached or self.cached_result is None:
            if self.edge_norm:
                edge_index, norm = GCNConv.norm(
                    edge_index, 
                    x.size(0), 
                    edge_weight, 
                    self.improved, 
                    x.dtype)
            else:
                norm = None
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        if self.edge_norm:
            return norm.view(-1, 1) * x_j
        else:
            return x_j
        
    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class CausalGAT(nn.Module):
    """
    Causal Attention Graph Neural Network for Vulnerability Detection.
    
    This model implements causal intervention by separating context and object features,
    then combining them through random shuffling during training to prevent spurious
    correlations.
    
    Args:
        num_features: Dimension of input graph node features
        num_classes: Number of output classes (2 for binary classification)
        tokens_dim: Dimension of code token embeddings (default: 768 for CodeT5)
        hidden_dim: Hidden layer dimension
        num_conv_layers: Number of GAT convolutional layers
        fusion_mode: How to fuse graph and token embeddings ('concat', 'gated', 'cross_atten', None)
        head: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(self, 
                 num_features: int,
                 num_classes: int,
                 tokens_dim: int = 768,
                 hidden_dim: int = 256,
                 num_conv_layers: int = 3,
                 fusion_mode: Optional[str] = None,
                 head: int = 4, 
                 dropout: float = 0.3):
        super(CausalGAT, self).__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "torch_geometric is required for CausalGAT. "
                "Install with: pip install torch_geometric torch_scatter"
            )
        
        self.cat_or_add = "add"
        self.fusion_mode = fusion_mode
        self.global_pool = global_add_pool
        self.dropout = dropout
        GConv = partial(GCNConv, edge_norm=True, gfn=False)

        hidden_in = num_features
        self.num_classes = num_classes
        hidden_out = num_classes
        self.fc_num = 222

        # Fusion layers based on mode
        if self.fusion_mode == 'concat':
            total_input_dim = num_features + tokens_dim
            self.concat_proj = Linear(total_input_dim, hidden_dim)
        elif self.fusion_mode == 'gated':
            self.feat_proj_gate = Linear(num_features, hidden_dim)
            self.code_proj_gate = Linear(tokens_dim, hidden_dim)
            self.gate_linear = Linear(num_features + tokens_dim, hidden_dim)
        elif self.fusion_mode == 'cross_atten':
            self.q_proj = Linear(num_features, hidden_dim)
            self.k_proj = Linear(tokens_dim, hidden_dim)
            self.v_proj = Linear(tokens_dim, hidden_dim)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, 
                num_heads=head, 
                batch_first=True
            )

        # Feature processing
        if self.fusion_mode is not None:
            self.bn_feat = BatchNorm1d(hidden_dim)
            self.conv_feat = GCNConv(hidden_dim, hidden_dim, gfn=True)
        else:
            self.bn_feat = BatchNorm1d(hidden_in)
            self.conv_feat = GCNConv(hidden_in, hidden_dim, gfn=True)
        
        # GAT convolution layers
        self.bns_conv = nn.ModuleList()
        self.convs = nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden_dim))
            self.convs.append(GATConv(hidden_dim, int(hidden_dim / head), heads=head, dropout=dropout))

        # Attention MLPs for causal intervention
        self.edge_att_mlp = Linear(hidden_dim * 2, 2)
        self.node_att_mlp = Linear(hidden_dim, 2)
        
        # Context and object convolutions
        self.bnc = BatchNorm1d(hidden_dim)
        self.bno = BatchNorm1d(hidden_dim)
        self.context_convs = GConv(hidden_dim, hidden_dim)
        self.objects_convs = GConv(hidden_dim, hidden_dim)

        # Context readout MLP
        self.fc1_bn_c = BatchNorm1d(hidden_dim)
        self.fc1_c = Linear(hidden_dim, hidden_dim)
        self.fc2_bn_c = BatchNorm1d(hidden_dim)
        self.fc2_c = Linear(hidden_dim, hidden_out)
        
        # Object readout MLP
        self.fc1_bn_o = BatchNorm1d(hidden_dim)
        self.fc1_o = Linear(hidden_dim, hidden_dim)
        self.fc2_bn_o = BatchNorm1d(hidden_dim)
        self.fc2_o = Linear(hidden_dim, hidden_out)
        
        # Combined readout MLP
        if self.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden_dim * 2)
            self.fc1_co = Linear(hidden_dim * 2, hidden_dim)
            self.fc2_bn_co = BatchNorm1d(hidden_dim)
            self.fc2_co = Linear(hidden_dim, hidden_out)
        elif self.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden_dim)
            self.fc1_co = Linear(hidden_dim, hidden_dim)
            self.fc2_bn_co = BatchNorm1d(hidden_dim)
            self.fc2_co = Linear(hidden_dim, hidden_out)
        else:
            raise ValueError(f"Unknown cat_or_add mode: {self.cat_or_add}")
        
        # BatchNorm initialization
        for m in self.modules():
            if isinstance(m, BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, eval_random: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            data: PyTorch Geometric Data object with:
                - x or feat: Node features (graph embeddings)
                - tokens: Token embeddings from code LM
                - edge_index: Graph connectivity
                - batch: Batch assignment for each node
            eval_random: Whether to shuffle context features (True during training)
            
        Returns:
            Tuple of (context_logits, object_logits, combined_logits)
        """
        x_features = data.x if data.x is not None else data.feat
        x_tokens = data.tokens if hasattr(data, 'tokens') and data.tokens is not None else x_features
        edge_index, batch = data.edge_index, data.batch
        row, col = edge_index

        # Feature fusion
        if self.fusion_mode == 'concat':
            x_concat = torch.cat([x_features, x_tokens], dim=-1)
            x = self.concat_proj(x_concat)
        elif self.fusion_mode == 'gated':
            x_feat_proj = self.feat_proj_gate(x_features)
            x_code_proj = self.code_proj_gate(x_tokens)
            gate_input = torch.cat([x_features, x_tokens], dim=-1)
            g = torch.sigmoid(self.gate_linear(gate_input))
            x = g * x_feat_proj + (1.0 - g) * x_code_proj
        elif self.fusion_mode == 'cross_atten':
            q = self.q_proj(x_features).unsqueeze(1)
            k = self.k_proj(x_tokens).unsqueeze(1)
            v = self.v_proj(x_tokens).unsqueeze(1)
            attn_output, _ = self.cross_attn(q, k, v)
            x = attn_output.squeeze(1)
        else:
            x = x_features
        
        # Initial feature processing
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        
        # GAT convolution layers
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index))
        
        # Edge and node attention for causal intervention
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]

        node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        xc = node_att[:, 0].view(-1, 1) * x
        xo = node_att[:, 1].view(-1, 1) * x
        
        # Context and object convolutions
        xc = F.relu(self.context_convs(self.bnc(xc), edge_index, edge_weight_c))
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index, edge_weight_o))

        # Global pooling
        xc = self.global_pool(xc, batch)
        xo = self.global_pool(xo, batch)
        
        # Readout layers
        xc_logis = self.context_readout_layer(xc)
        xo_logis = self.objects_readout_layer(xo)
        xco_logis = self.random_readout_layer(xc, xo, eval_random=eval_random)
        
        return xc_logis, xo_logis, xco_logis

    def context_readout_layer(self, x: torch.Tensor) -> torch.Tensor:
        """Context feature readout."""
        x = self.fc1_bn_c(x)
        x = self.fc1_c(x)
        x = F.relu(x)
        x = self.fc2_bn_c(x)
        x = self.fc2_c(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def objects_readout_layer(self, x: torch.Tensor) -> torch.Tensor:
        """Object feature readout."""
        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)
        x = self.fc2_o(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def random_readout_layer(self, xc: torch.Tensor, xo: torch.Tensor, eval_random: bool) -> torch.Tensor:
        """
        Combined readout with optional random shuffling of context features.
        
        This implements the causal intervention: by randomly shuffling the context,
        we break spurious correlations between context and labels during training.
        """
        num = xc.shape[0]
        l = [i for i in range(num)]
        if eval_random:
            random.shuffle(l)
        random_idx = torch.tensor(l, device=xc.device)
        
        if self.cat_or_add == "cat":
            x = torch.cat((xc[random_idx], xo), dim=1)
        else:
            x = xc[random_idx] + xo

        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.relu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis
    
    def predict(self, data) -> Tuple[int, float]:
        """
        Make a prediction for a single sample.
        
        Returns:
            Tuple of (predicted_class, confidence)
        """
        self.eval()
        with torch.no_grad():
            _, xo_logis, _ = self.forward(data, eval_random=False)
            probs = torch.exp(xo_logis)  # Convert log probs to probs
            pred_class = probs.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
        return pred_class, confidence


def is_available() -> bool:
    """Check if the required dependencies are available."""
    return TORCH_GEOMETRIC_AVAILABLE


