from typing import Tuple

import torch
import torch.nn as nn
from ding.utils import MODEL_REGISTRY

from .common import RepresentationNetwork


# use ModelRegistry to register the model, for more details about ModelRegistry, please refer to DI-engine's document.
@MODEL_REGISTRY.register('AlphaZXModel')
class AlphaZXModel(nn.Module):

    def __init__(self, num_of_sampled_actions: int = 50):
        """
        Overview:
            The definition of AlphaZX model, which is a general model for AlphaZX algorithm.
        """
        super(AlphaZXModel, self).__init__()
        self.num_of_sampled_actions = num_of_sampled_actions

        self.prediction_network = PredictionNetwork()

        if self.representation_network is None:
            self.representation_network = RepresentationNetwork()
        else:
            self.representation_network = self.representation_network

    def forward(self, state_batch: torch.Tensor) -> Tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Overview:
            The common computation graph of AlphaZero model.
        Arguments:
            - state_batch (:obj:`torch.Tensor`): The input state data, e.g. 2D image with the shape of [C, H, W].
        Returns:
            - policy_parameters (:obj:`torch.Tensor`): The output probability to select discrete action.
            - value (:obj:`torch.Tensor`): The output value of input state to help policy improvement and evaluation.
        Shapes:
            - state_batch (:obj:`torch.Tensor`): :math:`(B, C, H, W)`, where B is batch size, C is channel, H is \
                height, W is width.
            - logit (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size, N is action space size.
            - value (:obj:`torch.Tensor`): :math:`(B, 1)`, where B is batch size.
        """
        pass

    def compute_policy_value(self, state_batch: torch.Tensor) -> Tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Overview:
            The computation graph of AlphaZero model to calculate action selection probability and value.
        Arguments:
            - state_batch (:obj:`torch.Tensor`): The input state data, e.g. 2D image with the shape of [C, H, W].
        Returns:
            - policy_parameters (:obj:`torch.Tensor`): The output probability to select discrete action.
            - value (:obj:`torch.Tensor`): The output value of input state to help policy improvement and evaluation.
        Shapes:
            - state_batch (:obj:`torch.Tensor`): :math:`(B, C, H, W)`, where B is batch size, C is channel, H is \
                height, W is width.
            - policy_parameters (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size, N is action space size.
            - value (:obj:`torch.Tensor`): :math:`(B, 1)`, where B is batch size.
        """
        pass

    def compute_logp_value(self, state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            The computation graph of AlphaZero model to calculate log probability and value.
        Arguments:
            - state_batch (:obj:`torch.Tensor`): The input state data, e.g. 2D image with the shape of [C, H, W].
        Returns:
            - log_prob (:obj:`torch.Tensor`): The output log probability to select discrete action.
            - value (:obj:`torch.Tensor`): The output value of input state to help policy improvement and evaluation.
        Shapes:
            - state_batch (:obj:`torch.Tensor`): :math:`(B, C, H, W)`, where B is batch size, C is channel, H is \
                height, W is width.
            - log_prob (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size, N is action space size.
            - value (:obj:`torch.Tensor`): :math:`(B, 1)`, where B is batch size.
        """
        pass


class PredictionNetwork(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Use the hidden state to predict the value and policy.
        Arguments:
            - x (:obj:`torch.Tensor`): The hidden state.
        Returns:
            - outputs (:obj:`Tuple[torch.Tensor, torch.Tensor]`): The value and policy.
        Shapes:
            - x (:obj:`torch.Tensor`): :math:`(B, C, H, W)`, where B is batch size, C is channel, H is \
                the height of the encoding state, W is width of the encoding state.
            - logit (:obj:`torch.Tensor`): :math:`(B, N)`, where B is batch size, N is action space size.
            - value (:obj:`torch.Tensor`): :math:`(B, 1)`, where B is batch size.
        """
        pass
