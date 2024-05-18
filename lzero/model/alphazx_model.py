from typing import Tuple, Any

import torch
import torch.nn as nn
import torch_geometric as pyg
from alphazx.distributions.alpha_zx_dist import AlphaZXDistributionParams
from alphazx.models.homogeneous.prediction_network import PredictionNetwork
from alphazx.models.homogeneous.representation_network import RepresentationNetwork
from ding.utils import MODEL_REGISTRY


@MODEL_REGISTRY.register('AlphaZXModel')
class AlphaZXModel(nn.Module):

    def __init__(self,
                 num_node_types: int,
                 num_possible_phases: int,
                 num_possible_new_edges: int,
                 repr_node_embedding_channels: int,
                 repr_gps_channels: int,
                 repr_gps_edge_in_channels: int,
                 repr_gps_edge_out_channels: int,
                 repr_gps_pe_in_channels: int,
                 repr_gps_pe_out_channels: int,
                 repr_gps_num_layers: int,
                 repr_gps_bias: bool,
                 repr_gps_num_attn_heads: int,
                 repr_gps_attn_type: str,
                 repr_gps_attn_kwargs: dict[str, Any],
                 repr_gps_mlp_hidden_channels: int,
                 policy_node_embedding_channels: int,
                 policy_gps_channels: int,
                 policy_gps_edge_in_channels: int,
                 policy_gps_edge_out_channels: int,
                 policy_gps_pe_in_channels: int,
                 policy_gps_pe_out_channels: int,
                 policy_gps_num_layers: int,
                 policy_gps_bias: bool,
                 policy_gps_num_attn_heads: int,
                 policy_gps_attn_type: str,
                 policy_gps_attn_kwargs: dict[str, Any],
                 policy_gps_mlp_hidden_channels: int,
                 policy_num_pooling_encoder_blocks: int,
                 policy_num_pooling_heads: int,
                 policy_pooling_layer_norm: bool,
                 policy_pooling_dropout: float,
                 value_node_embedding_channels: int,
                 value_gps_channels: int,
                 value_gps_edge_in_channels: int,
                 value_gps_edge_out_channels: int,
                 value_gps_pe_in_channels: int,
                 value_gps_pe_out_channels: int,
                 value_gps_num_layers: int,
                 value_gps_bias: bool,
                 value_gps_num_attn_heads: int,
                 value_gps_attn_type: str,
                 value_gps_attn_kwargs: dict[str, Any],
                 value_gps_mlp_hidden_channels: int,
                 value_gmt_num_encoder_blocks: int,
                 value_gmt_num_heads: int,
                 value_gmt_layer_norm: bool,
                 value_gmt_dropout: float):
        super(AlphaZXModel, self).__init__()
        self.representation_network = RepresentationNetwork(num_node_types,
                                                            num_possible_phases,
                                                            repr_node_embedding_channels,
                                                            repr_gps_channels,
                                                            repr_gps_edge_in_channels,
                                                            repr_gps_edge_out_channels,
                                                            repr_gps_pe_in_channels,
                                                            repr_gps_pe_out_channels,
                                                            repr_gps_num_layers,
                                                            repr_gps_bias,
                                                            repr_gps_num_attn_heads,
                                                            repr_gps_attn_type,
                                                            repr_gps_attn_kwargs,
                                                            repr_gps_mlp_hidden_channels)
        self.prediction_network = PredictionNetwork(num_node_types,
                                                    num_possible_phases,
                                                    num_possible_new_edges,
                                                    policy_node_embedding_channels,
                                                    policy_gps_channels,
                                                    policy_gps_edge_in_channels,
                                                    policy_gps_edge_out_channels,
                                                    policy_gps_pe_in_channels,
                                                    policy_gps_pe_out_channels,
                                                    policy_gps_num_layers,
                                                    policy_gps_bias,
                                                    policy_gps_num_attn_heads,
                                                    policy_gps_attn_type,
                                                    policy_gps_attn_kwargs,
                                                    policy_gps_mlp_hidden_channels,
                                                    policy_num_pooling_encoder_blocks,
                                                    policy_num_pooling_heads,
                                                    policy_pooling_layer_norm,
                                                    policy_pooling_dropout,
                                                    value_node_embedding_channels,
                                                    value_gps_channels,
                                                    value_gps_edge_in_channels,
                                                    value_gps_edge_out_channels,
                                                    value_gps_pe_in_channels,
                                                    value_gps_pe_out_channels,
                                                    value_gps_num_layers,
                                                    value_gps_bias,
                                                    value_gps_num_attn_heads,
                                                    value_gps_attn_type,
                                                    value_gps_attn_kwargs,
                                                    value_gps_mlp_hidden_channels,
                                                    value_gmt_num_encoder_blocks,
                                                    value_gmt_num_heads,
                                                    value_gmt_layer_norm,
                                                    value_gmt_dropout)

    def forward(self, data: pyg.data.Data) -> Tuple[AlphaZXDistributionParams, torch.Tensor]:
        data.x = self.representation_network(data)
        policy, value = self.prediction_network(data)
        return policy, value

    def compute_policy_value(self, data: pyg.data.Data) -> Tuple[AlphaZXDistributionParams, torch.Tensor]:
        data.x = self.representation_network(data)
        policy, value = self.prediction_network(data)
        return policy, value

    def compute_logp_value(self, data: pyg.data.Data) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
