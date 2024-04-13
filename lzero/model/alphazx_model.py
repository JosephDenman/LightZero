from typing import Tuple, Dict, Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from alphazx.diagram.match import METADATA, NODE_METADATA
from ding.utils import MODEL_REGISTRY
from torch_geometric.nn import HGTConv
from torch_geometric.typing import NodeType, Metadata, EdgeType


def stack_feat_dict(x_dict: Dict[NodeType, torch.Tensor], include_offsets: bool = True) -> Union[
    torch.Tensor, Tuple[torch.Tensor, Dict[NodeType, int]]]:
    """Stacks a dictionary of features."""
    cum_sum = 0
    t = []
    offsets = {}
    for key, x in x_dict.items():
        t.append(x)
        offsets[key] = cum_sum
        cum_sum += x.size(0)
    t = torch.stack(t, dim=0)
    return t, offsets if include_offsets else t


def unstack_feat_dict(x: torch.Tensor, offsets: Dict[NodeType, int]) -> Dict[NodeType, torch.Tensor]:
    pass


class NodeFeedForward(nn.Module):
    def __init__(self,
                 input_dim: int,
                 feedforward_dim: int,
                 dropout: float = 0.1,
                 act: str = 'relu',
                 layer_norm_eps: float = 1e-5,
                 bias: bool = True,
                 norm_first: bool = False):
        super(NodeFeedForward, self).__init__()
        self.linear1 = nn.Linear(input_dim, feedforward_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, input_dim, bias=bias)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(input_dim, eps=layer_norm_eps, bias=bias)
        self.norm2 = nn.LayerNorm(input_dim, eps=layer_norm_eps, bias=bias)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if act == 'relu':
            self.act = F.relu
        elif act == 'gelu':
            self.act = F.gelu

    def _feed_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_first:
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm2(x + self._ff_block(x))
        return x


class NodeWiseFeedForward(nn.Module):
    def __init__(self,
                 input_dim: int,
                 feedforward_dim: int,
                 dropout: float = 0.1,
                 act: str = 'relu',
                 layer_norm_eps: float = 1e-5,
                 bias: bool = True,
                 norm_first: bool = False):
        super(NodeWiseFeedForward, self).__init__()
        self.mlp_dict = nn.ModuleDict({
            node_type: NodeFeedForward(input_dim, feedforward_dim, dropout, act, layer_norm_eps, bias, norm_first) for
            node_type in NODE_METADATA
        })

    def forward(self, x_dict: Dict[NodeType, torch.Tensor]) -> Dict[NodeType, torch.Tensor]:
        for node_type in NODE_METADATA:
            x_dict[node_type] = self.mlp_dict[node_type](x_dict[node_type])
        return x_dict


class NodeEncoder(nn.Module):
    def __init__(self,
                 num_blocks: int,
                 input_dim: int,
                 attn_heads: int,
                 feedforward_dim: int = 2048,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 layer_norm_eps: float = 1e-5,
                 bias: bool = True,
                 norm_first: bool = False):
        super(NodeEncoder, self).__init__()
        self.node_encoder = NodeEncoder(num_blocks, input_dim, attn_heads, feedforward_dim, dropout, activation,
                                        layer_norm_eps, bias)
        self.node_wise_ff = NodeWiseFeedForward(input_dim, feedforward_dim, dropout, activation, layer_norm_eps, bias,
                                                norm_first)

    def forward(self, x_dict: Dict[NodeType, torch.Tensor]) -> Dict[NodeType, torch.Tensor]:
        x, offsets = stack_feat_dict(x_dict)
        x = self.node_encoder(x)
        x_dict = unstack_feat_dict(x, offsets)
        x_dict = self.node_wise_ff(x_dict)
        return x_dict


class HGT(torch.nn.Module):
    def __init__(self,
                 metadata: Metadata,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_heads: int,
                 num_layers: int):
        super(HGT, self).__init__()
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = pyg.nn.Linear(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            self.convs.append(conv)
        self.lin = pyg.nn.Linear(hidden_channels, out_channels)

    def forward(self,
                x_dict: Dict[NodeType, torch.Tensor],
                edge_index_dict: Dict[EdgeType, torch.Tensor]) -> Dict[NodeType, torch.Tensor]:
        x_dict = {node_type: self.lin_dict[node_type](x).relu_() for node_type, x in x_dict.items()}
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
        return self.lin(x_dict['author'])


class RepresentationNetwork(nn.Module):

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_heads: int,
                 num_layers: int) -> None:
        super(RepresentationNetwork, self).__init__()
        self.hgt = HGT(METADATA, in_channels, hidden_channels, out_channels, num_heads, num_layers)

    def forward(self,
                x_dict: Dict[NodeType, torch.Tensor],
                edge_index_dict: Dict[EdgeType, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.hgt(x_dict, edge_index_dict)


# TODO: Move this to 'alphazx.distribution'
AZXDistParams = Dict[Union[Literal['flz_node_dist_params'], Literal['frz_node_dist_params']], torch.Tensor]


class ValueNetwork(nn.Module):
    def __init__(self,
                 hgt_in_dim: int,
                 hgt_hidden_dim: int,
                 hgt_out_dim: int,
                 hgt_heads: int,
                 hgt_layers: int,
                 self_attn_blocks: int,
                 self_attn_out_dim: int,
                 self_attn_heads: int,
                 pooling_encoder_blocks: int,
                 pooling_heads: int,
                 pooling_layer_norm: bool = False,
                 pooling_dropout: float = 0.0):
        super(ValueNetwork, self).__init__()
        self.hgt = HGT(METADATA, hgt_in_dim, hgt_hidden_dim, hgt_out_dim, hgt_heads, hgt_layers)
        self.self_attn = GlobalSelfAttention(self_attn_blocks, hgt_out_dim, self_attn_out_dim, self_attn_heads)
        self.pool = pyg.nn.GraphMultisetTransformer(self_attn_out_dim, 1, pooling_encoder_blocks, pooling_heads,
                                                    pooling_layer_norm, pooling_dropout)

    def forward(self,
                x_dict: Dict[NodeType, torch.Tensor],
                edge_index_dict: Dict[EdgeType, torch.Tensor]) -> torch.Tensor:
        x_dict = self.hgt(x_dict, edge_index_dict)
        x_dict = self.self_attn(x_dict, edge_index_dict)
        # TODO: The next two assignments are probably inefficient.
        data = pyg.data.HeteroData(x_dict.update(edge_index_dict)).to_homogeneous(node_attrs=['phase'],
                                                                                  add_node_type=True,
                                                                                  add_edge_type=True)
        data = data.sort(False)
        h = self.pool(data.x, data.edge_index)
        return h


# TODO: Note summation aggregations are useful for structural characteristics. Non-permutation invariance aggregators
#       typically work well when the input is in some canonical form, which is not the case for us.
class PredictionNetwork(nn.Module):
    def __init__(self,
                 input_dim: int,
                 embed_dim: int,
                 hgt_hidden_dim: int,
                 hgt_heads: int,
                 hgt_layers: int,
                 value_hgt_hidden_dim: int,
                 value_out_dim: int,
                 value_hgt_heads: int,
                 value_hgt_layers: int,
                 value_self_attn_blocks: int,
                 value_self_attn_out_dim: int,
                 value_self_attn_heads: int,
                 value_pooling_encoder_blocks: int,
                 value_pooling_heads: int,
                 value_pooling_layer_norm: bool,
                 value_pooling_dropout: float,
                 self_attn_blocks: int,
                 self_attn_heads: int,
                 mlp_hidden_dim: int,
                 mlp_num_layers: int,
                 mlp_act: str = 'relu'):
        super(PredictionNetwork, self).__init__()
        self.hgt = HGT(METADATA, input_dim, hgt_hidden_dim, embed_dim, hgt_heads, hgt_layers)
        self.value_network = ValueNetwork(embed_dim, value_hgt_hidden_dim, value_out_dim, value_hgt_heads,
                                          value_hgt_layers, value_self_attn_blocks, value_self_attn_out_dim,
                                          value_self_attn_heads, value_pooling_encoder_blocks, value_pooling_heads,
                                          value_pooling_layer_norm, value_pooling_dropout)
        self.global_self_attn_0 = GlobalSelfAttention(self_attn_blocks, embed_dim, embed_dim, self_attn_heads)
        self.node_mlp_dict_0 = NodeMLPDict(embed_dim, mlp_hidden_dim, embed_dim, mlp_num_layers, mlp_act)
        self.global_self_attn_1 = GlobalSelfAttention(self_attn_blocks, embed_dim, embed_dim, self_attn_heads)

    def forward(self,
                x_dict: Dict[NodeType, torch.Tensor],
                edge_index_dict: Dict[EdgeType, torch.Tensor]) -> Tuple[AZXDistParams, torch.Tensor]:
        value = self.value_network(x_dict, edge_index_dict)
        x_dict = self.global_self_attn_0(x_dict, edge_index_dict)
        for node_type in NODE_METADATA:
            x_dict[node_type] = self.mlp_dict[node_type](x_dict[node_type])
        x_dict = self.global_self_attn_1(x_dict, edge_index_dict)
        return x_dict, value


@MODEL_REGISTRY.register('AlphaZXModel')
class AlphaZXModel(nn.Module):

    def __init__(self,
                 input_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 hidden_channels: int,
                 num_layers: int,
                 act: str = 'relu'):
        super(AlphaZXModel, self).__init__()
        self.representation_network = RepresentationNetwork()
        self.prediction_network = PredictionNetwork(input_dim, embed_dim, num_heads, hidden_channels, num_layers, act)

    def forward(self,
                x_dict: Dict[NodeType, torch.Tensor],
                edge_index_dict: Dict[EdgeType, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        x_dict = self.representation_network(x_dict, edge_index_dict)
        policy, value = self.prediction_network(x_dict, edge_index_dict)
        return policy, value

    def compute_policy_value(self, state_batch: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        pass

    def compute_logp_value(self, state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
