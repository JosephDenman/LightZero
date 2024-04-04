import copy
from collections import namedtuple
from typing import List, Dict, Tuple, Any

import torch.distributions
import torch.nn.functional as F
import torch.optim as optim
from ding.policy.base_policy import Policy
from ding.torch_utils import to_device
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate
from src.distributions.alpha_zx_dist import AlphaZXDistribution

from lzero.policy import configure_optimizers
from lzero.policy.utils import pad_and_get_lengths
from zoo.graphs.config.alphazx_config import alphazx_config
from zoo.graphs.envs.alphazx_env import AlphaZXEnv


@POLICY_REGISTRY.register('alphazx')
class AlphaZXPolicy(Policy):
    """
    Overview:
        The policy class for AlphaZX.
    """

    # The default_config for AlphaZX policy.
    config = dict(
        # (str) The type of policy, as the key of the policy registry.
        type='alphazx',
        # this variable is used in ``collector``.
        normalize_prob_of_sampled_actions=False,
        policy_loss_type='cross_entropy',  # options={'cross_entropy', 'KL'}
        # (bool) Whether to use torch.compile method to speed up our model, which required torch>=2.0.
        torch_compile=False,
        # (bool) Whether to use TF32 for our model.
        tensor_float_32=False,
        model=dict(
            # (int) The number of channels of hidden states in AlphaZX model.
            num_channels=32,
        ),
        # (bool) Whether to use C++ MCTS in policy. If False, use Python implementation.
        mcts_ctree=True,
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (int) How many updates(iterations) to train after collector's one collection.
        # Bigger "update_per_collect" means bigger off-policy.
        # collect data -> update policy-> collect data -> ...
        # For different env, we have different episode_length,
        # we usually set update_per_collect = collector_env_num * episode_length / batch_size * reuse_factor.
        # If we set update_per_collect=None, we will set update_per_collect = collected_transitions_num * cfg.policy.model_update_ratio automatically.
        update_per_collect=None,
        # (float) The ratio of the collected data used for training. Only effective when ``update_per_collect`` is not None.
        model_update_ratio=0.1,
        # (int) Minibatch size for one gradient descent.
        batch_size=256,
        # (str) Optimizer for training policy network. ['SGD', 'Adam', 'AdamW']
        optim_type='SGD',
        # (float) Learning rate for training policy network. Initial lr for manually decay schedule.
        learning_rate=0.2,
        # (float) Weight decay for training policy network.
        weight_decay=1e-4,
        # (float) One-order Momentum in optimizer, which stabilizes the training process (gradient direction).
        momentum=0.9,
        # (float) The maximum constraint value of gradient norm clipping.
        grad_clip_value=10,
        # (float) The weight of value loss.
        value_weight=1.0,
        # (int) The number of environments used in collecting data.
        collector_env_num=8,
        # (int) The number of environments used in evaluating policy.
        evaluator_env_num=3,
        # (bool) Whether to use piecewise constant learning rate decay.
        # i.e. lr: 0.2 -> 0.02 -> 0.002
        lr_piecewise_constant_decay=True,
        # (int) The number of final training iterations to control lr decay, which is only used for manually decay.
        threshold_training_steps_for_final_lr=int(5e5),
        # (bool) Whether to use manually temperature decay.
        # i.e. temperature: 1 -> 0.5 -> 0.25
        manual_temperature_decay=False,
        # (int) The number of final training iterations to control temperature, which is only used for manually decay.
        threshold_training_steps_for_final_temperature=int(1e5),
        # (float) The fixed temperature value for MCTS action selection, which is used to control the exploration.
        # The larger the value, the more exploration. This value is only used when manual_temperature_decay=False.
        fixed_temperature_value=0.25,
        mcts=dict(
            # (int) The number of simulations to perform at each move.
            num_simulations=50,
            # (int) The maximum number of moves to make in a game.
            max_moves=512,  # for chess and shogi, 722 for Go.
            # (float) The alpha value used in the Dirichlet distribution for exploration at the root node of the search tree.
            root_dirichlet_alpha=0.3,
            # (float) The noise weight at the root node of the search tree.
            root_noise_weight=0.25,
            # (int) The base constant used in the PUCT formula for balancing exploration and exploitation during tree search.
            pb_c_base=19652,
            # (float) The initialization constant used in the PUCT formula for balancing exploration and exploitation during tree search.
            pb_c_init=1.25,
            # (int) The action space size.
            min_action_size=9,
            # (int) The number of sampled actions for each state.
            num_of_sampled_actions=50
        ),
        other=dict(replay_buffer=dict(
            replay_buffer_size=int(1e6),
            save_episode=False,
        )),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default model setting for demonstration.
        Returns:
            - model_type (:obj:`str`): The model type used in this algorithm, which is registered in ModelRegistry.
            - import_names (:obj:`List[str]`): The model class path list used in this algorithm.
        """
        return 'AlphaZXModel', ['lzero.model.alphazx_model']

    def _init_learn(self) -> None:
        assert self._cfg.optim_type in ['SGD', 'Adam', 'AdamW'], self._cfg.optim_type
        if self._cfg.optim_type == 'SGD':
            self._optimizer = optim.SGD(
                self._model.parameters(),
                lr=self._cfg.learning_rate,
                momentum=self._cfg.momentum,
                weight_decay=self._cfg.weight_decay,
            )
        elif self._cfg.optim_type == 'Adam':
            self._optimizer = optim.Adam(
                self._model.parameters(), lr=self._cfg.learning_rate, weight_decay=self._cfg.weight_decay
            )
        elif self._cfg.optim_type == 'AdamW':
            self._optimizer = configure_optimizers(
                model=self._model,
                weight_decay=self._cfg.weight_decay,
                learning_rate=self._cfg.learning_rate,
                device_type=self._cfg.device
            )

        if self._cfg.lr_piecewise_constant_decay:
            from torch.optim.lr_scheduler import LambdaLR
            max_step = self._cfg.threshold_training_steps_for_final_lr
            # NOTE: the 1, 0.1, 0.01 is the decay rate, not the lr.
            # lr_lambda = lambda step: 1 if step < max_step * 0.5 else (0.1 if step < max_step else 0.01)  # noqa
            lr_lambda = lambda step: 1 if step < max_step * 0.33 else (0.1 if step < max_step * 0.66 else 0.01)  # noqa
            self.lr_scheduler = LambdaLR(self._optimizer, lr_lambda=lr_lambda)

        # Algorithm config
        self._value_weight = self._cfg.value_weight
        self._entropy_weight = self._cfg.entropy_weight
        # Main and target models
        self._learn_model = self._model

        # TODO(pu): test the effect of torch 2.0
        if self._cfg.torch_compile:
            self._learn_model = torch.compile(self._learn_model)

    def _forward_learn(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        # list of dict -> dict of list
        # only for env with variable legal actions
        inputs = pad_and_get_lengths(inputs, self._cfg.mcts.num_of_sampled_actions)
        inputs = default_collate(inputs)

        if self._cuda:
            inputs = to_device(inputs, self._device)
        self._learn_model.train()

        state_batch = inputs['obs']['observation']
        # The output of 'get_next_action' in 'ptree_alphazx.py'
        mcts_visit_count_probs = inputs['probs']
        reward = inputs['reward']
        root_sampled_actions = inputs['root_sampled_actions']

        root_sampled_actions = root_sampled_actions.to(device=self._device, dtype=torch.float)
        state_batch = state_batch.to(device=self._device, dtype=torch.float)
        mcts_visit_count_probs = mcts_visit_count_probs.to(device=self._device, dtype=torch.float)
        reward = reward.to(device=self._device, dtype=torch.float)

        # TODO: 'policy_params' is a dictionary of tensors mapping sets of policy parameters to their values. For example,
        #       'policy_params' contains 'mixture_params' that maps to the tensor of mixture parameters for each batch.
        policy_params, values = self._learn_model.compute_policy_value(state_batch)

        azx_dist = AlphaZXDistribution(policy_params)
        policy_entropy_loss = -azx_dist.entropy()

        # ==============================================================
        # policy loss
        # ==============================================================
        policy_loss = self._calculate_policy_loss_disc(azx_dist, mcts_visit_count_probs, root_sampled_actions)

        # ==============================================================
        # value loss
        # ==============================================================
        value_loss = F.mse_loss(values.view(-1), reward)

        total_loss = self._value_weight * value_loss + policy_loss + self._entropy_weight * policy_entropy_loss
        self._optimizer.zero_grad()
        total_loss.backward()

        total_grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
            list(self._model.parameters()),
            max_norm=self._cfg.grad_clip_value,
        )
        self._optimizer.step()
        if self._cfg.lr_piecewise_constant_decay is True:
            self.lr_scheduler.step()

        # =============
        # after update
        # =============
        return {
            'cur_lr': self._optimizer.param_groups[0]['lr'],
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'policy_entropy_loss': policy_entropy_loss.item(),
            'total_grad_norm_before_clip': total_grad_norm_before_clip.item(),
            'collect_mcts_temperature': self.collect_mcts_temperature,
        }

    def _calculate_policy_loss_disc(self,
                                    azx_dist: AlphaZXDistribution,
                                    mcts_policy: torch.Tensor,
                                    root_sampled_actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            azx_dist: The policy parameters (in continuous spaces) or policy probabilities (in discrete spaces) predicted
                          by the model.
            mcts_policy: The empirical policy probabilities obtained from MCTS.
            root_sampled_actions: The actions sampled during MCTS.

        Returns:

        """

        # For each batch and sampled action, get the corresponding probability from 'AlphaZXDistribution'. This is only
        # valid if each entry in 'root_sampled_actions' corresponds to the correct entry in 'empirical_probs'.
        model_policy = azx_dist.prob(root_sampled_actions)
        # Normalize empirical_probs
        mcts_policy = mcts_policy / (mcts_policy.sum(dim=1, keepdim=True) + 1e-6)

        if self._cfg.policy_loss_type == 'KL':
            # Calculate the KL divergence between policy_probs and sampled_target_policy
            # The KL divergence between 2 probability distributions P and Q is defined as:
            # KL(P || Q) = sum(P(i) * log(P(i) / Q(i)))
            # We use the PyTorch function kl_div to calculate it.
            loss = torch.nn.functional.kl_div(
                model_policy.log(), mcts_policy, reduction='none'
            )
        elif self._cfg.policy_loss_type == 'cross_entropy':
            # Calculate the cross entropy loss between policy_probs and sampled_target_policy
            # The cross entropy between 2 probability distributions P and Q is defined as:
            # H(P, Q) = -sum(P(i) * log(Q(i)))
            # We use the PyTorch function cross_entropy to calculate it.
            loss = torch.nn.functional.cross_entropy(
                model_policy, torch.argmax(mcts_policy, dim=1), reduction='none'
            )
        else:
            raise ValueError(f"Invalid policy_loss_type: {self._cfg.policy_loss_type}")
        # 使用 nan_to_num 将 loss 中的 nan 值设置为0
        loss = torch.nan_to_num(loss)
        # Calculate the mean loss over the batch
        loss = loss.sum()
        return loss

    def _init_collect(self) -> None:
        """
        Overview:
            Collect mode init method. Called by ``self.__init__``. Initialize the collect model and MCTS utils.
        """
        self._get_simulation_env()

        self._collect_model = self._model
        if self._cfg.mcts_ctree:
            import sys
            sys.path.append('./LightZero/lzero/mcts/ctree/ctree_alphazx/build')
            import mcts_alphazx
            self._collect_mcts = mcts_alphazx.MCTS(self._cfg.mcts.max_moves, self._cfg.mcts.num_simulations,
                                                   self._cfg.mcts.pb_c_base,
                                                   self._cfg.mcts.pb_c_init, self._cfg.mcts.root_dirichlet_alpha,
                                                   self._cfg.mcts.root_noise_weight, self.simulate_env)
        else:
            from lzero.mcts.ptree.ptree_alphazx import MCTS
            self._collect_mcts = MCTS(self._cfg.mcts, self.simulate_env)
        self.collect_mcts_temperature = 1

    @torch.no_grad()
    def _forward_collect(self, obs: Dict, temperature: float = 1) -> Dict[str, Any]:

        """
        Overview:
            The forward function for collecting data in collect mode. Use real env to execute MCTS search.
        Arguments:
            - obs (:obj:`Dict`): The dict of obs, the key is env_id and the value is the \
                corresponding obs in this timestep.
            - temperature (:obj:`float`): The temperature for MCTS search.
        Returns:
            - output (:obj:`Dict[str, torch.Tensor]`): The dict of output, the key is env_id and the value is the \
                the corresponding policy output in this timestep, including action, probs and so on.
        """
        self.collect_mcts_temperature = temperature
        ready_env_id = list(obs.keys())
        output = {}
        self._policy_model = self._collect_model
        for env_id in ready_env_id:
            action, mcts_visit_count_probs = self._collect_mcts.get_next_action(
                self._policy_value_func,
                self.collect_mcts_temperature,
                True,
            )
            # These don't need to be converted into tensors because the collector does it.
            output[env_id] = {
                'action': action,
                'probs': mcts_visit_count_probs,
                'root_sampled_actions': self._collect_mcts.get_sampled_actions(),
            }
        return output

    def _init_eval(self) -> None:
        """
        Overview:
            Evaluate mode init method. Called by ``self.__init__``. Initialize the eval model and MCTS utils.
        """
        self._get_simulation_env()
        if self._cfg.mcts_ctree:
            import sys
            sys.path.append('./LightZero/lzero/mcts/ctree/ctree_alphazero/build')
            import mcts_alphazx
            # TODO(pu): how to set proper num_simulations for evaluation
            self._eval_mcts = mcts_alphazx.MCTS(self._cfg.mcts.max_moves,
                                                min(800, self._cfg.mcts.num_simulations * 4),
                                                self._cfg.mcts.pb_c_base,
                                                self._cfg.mcts.pb_c_init, self._cfg.mcts.root_dirichlet_alpha,
                                                self._cfg.mcts.root_noise_weight, self.simulate_env)
        else:
            from lzero.mcts.ptree.ptree_alphazx import MCTS
            mcts_eval_config = copy.deepcopy(self._cfg.mcts)
            # TODO(pu): how to set proper num_simulations for evaluation
            mcts_eval_config.num_simulations = min(800, mcts_eval_config.num_simulations * 4)
            self._eval_mcts = MCTS(mcts_eval_config, self.simulate_env)

        self._eval_model = self._model

    def _forward_eval(self, obs: Dict) -> Dict[str, Any]:

        """
        Overview:
            The forward function for evaluating the current policy in eval mode, similar to ``self._forward_collect``.
        Arguments:
            - obs (:obj:`Dict`): The dict of obs, the key is env_id and the value is the \
                corresponding obs in this timestep.
        Returns:
            - output (:obj:`Dict[str, torch.Tensor]`): The dict of output, the key is env_id and the value is the \
                the corresponding policy output in this timestep, including action, probs and so on.
        """
        ready_env_id = list(obs.keys())
        output = {}
        self._policy_model = self._eval_model
        for env_id in ready_env_id:
            action, mcts_visit_count_probs = self._eval_mcts.get_next_action(self._policy_value_func, 1.0, False)
            output[env_id] = {
                'action': action,
                'probs': mcts_visit_count_probs,
            }
        return output

    def _get_simulation_env(self):
        self.simulate_env = AlphaZXEnv(alphazx_config)

    @torch.no_grad()
    def _policy_value_func(self, environment: 'Environment') -> tuple[list[list], float]:
        # Retrieve the current state and its scale from the environment
        current_state, state_scale = environment.current_state()

        # Convert the state scale to a PyTorch FloatTensor, adding a dimension to match the model's input requirements
        state_scale_tensor = torch.from_numpy(state_scale).to(
            device=self._device, dtype=torch.float
        ).unsqueeze(0)

        # Compute policy parameters and state value for the current state using the policy model, without gradient computation
        with torch.no_grad():
            policy_parameters, state_value = self._policy_model.compute_policy_value(state_scale_tensor)
        sampled_actions = AlphaZXDistribution(policy_parameters).sample(self._cfg.mcts.num_of_sampled_actions).tolist()
        return sampled_actions, state_value.item()

    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Register the variables to be monitored in learn mode. The registered variables will be logged in
            tensorboard according to the return value ``_forward_learn``.
        """
        return super()._monitor_vars_learn() + [
            'cur_lr', 'total_loss', 'policy_loss', 'value_loss', 'policy_entropy_loss', 'total_grad_norm_before_clip',
            'collect_mcts_temperature'
        ]

    def _process_transition(self, obs: Dict, model_output: Dict[str, torch.Tensor], timestep: namedtuple) -> Dict:
        """
        Overview:
            Generate the dict type transition (one timestep) data from policy learning.
        """
        return {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': model_output['action'],
            'root_sampled_actions': model_output['root_sampled_actions'],
            'probs': model_output['probs'],
            'reward': timestep.reward,
            'done': timestep.done,
        }

    def _get_train_sample(self, data):
        # be compatible with DI-engine Policy class
        pass
