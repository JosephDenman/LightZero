import copy
from typing import Any, Optional, Tuple

import networkx as nx
from alphazx.diagram.feature_conversions import cat_phase_to_float, cat_new_edges_to_int, \
    bernoulli_transfer_edges_to_set
from alphazx.diagram.match import Match, FRightXMatch, FLeftXMatch, BRightMatch, BLeftMatch, YRightZMatch, YLeftZMatch, \
    YRightXMatch, YLeftXMatch, FLeftZMatch, FRightZMatch
from ding.envs import BaseEnv
from ding.utils import ENV_REGISTRY
from easydict import EasyDict
from alphazx.diagram.diagram_generators import clifford_zx_diagram
from alphazx.diagram.zx_match_diagram import to_zx_match_diagram, HeteroDataIndexToMatch
from alphazx.game.zx_game import diagram_value, assert_correct_match_instance
from alphazx.rewriting.util import rewrite, FRightParameters


@ENV_REGISTRY.register('alphazx')
class AlphaZXEnv(BaseEnv):
    """
    Overview:
        An AlphaZX environment that inherits from the BaseEnv. This environment can be used for training and
        evaluating AI players for the game of Gomoku.

    .. note::
        For the latest macOS, you should set context='spawn' or 'forkserver' in ding/envs/env_manager/subprocess_env_manager.py
        to be able to use subprocess env_manager.
    """

    config = dict(
        # (str) The name of the environment registered in the environment registry.
        env_id="alphazx",
        # (int) The maximum number of qubits in the initial graph.
        max_qubits=50,
        # (int) The maximum gate depth of the initial graph.
        max_depth=50,
        # (str) The mode of the environment when take a step.
        battle_mode='self_play_mode',
        # (str) The mode of the environment when doing the MCTS.
        battle_mode_in_simulation_env='self_play_mode',  # only used in AlphaZero
        # (str) The render mode. Options are 'None', 'state_realtime_mode', 'image_realtime_mode' or 'image_savefile_mode'.
        # If None, then the game will not be rendered.
        render_mode=None,
        # (str or None) The directory in which to save the replay file. If None, the file is saved in the current directory.
        replay_path=None,
        # (float) The scale of the render screen.
        screen_scaling=9,
        # (bool) Whether to use the 'channel last' format for the observation space. If False, 'channel first' format is used.
        channel_last=False,
        # (bool) Whether to scale the observation.
        scale=True,
        # (bool) Whether to let human to play with the agent when evaluating. If False, then use the bot to evaluate the agent.
        agent_vs_human=False,
        # (float) The probability that a random agent is used instead of the learning agent.
        prob_random_agent=0,
        # (float) The probability that a random action will be taken when calling the bot.
        prob_random_action_in_bot=0.,
        # (bool) Whether to use the MCTS ctree in AlphaZX. If True, then the AlphaZero MCTS ctree will be used.
        alphazx_mcts_ctree=False,
    )

    done = False

    def __init__(self, cfg: dict = None):
        self._cfg = cfg
        self.num_qubits = cfg.num_qubits
        self.depth = cfg.depth
        self.t_gates = cfg.t_gates
        self.done_reward = cfg.done_reward
        self.step_penalty = cfg.step_penalty
        self.zx_diagram = clifford_zx_diagram(self.num_qubits, self.depth, self.t_gates)
        self.zx_match_diagram = to_zx_match_diagram(self.zx_diagram)
        self.hdata_node_index: Optional[HeteroDataIndexToMatch] = None
        self.previous_value = diagram_value(self.zx_diagram)
        self.episode_return = 0
        self.episode_length = 0
        self.done = False
        self.previous_reward = 0

    def __remove_isolated_nodes(self) -> None:
        self.zx_diagram.remove_nodes_from(list(nx.isolates(self.zx_diagram)))

    def __remove_self_loop_edges(self) -> None:
        self.zx_diagram.remove_edges_from(list(nx.selfloop_edges(self.zx_diagram, keys=True)))

    def __remove_isolated_components(self) -> None:
        if self.zx_diagram.num_b_nodes() == 0 or self.zx_diagram.num_b_nodes() == 1:
            raise ValueError('Valid diagrams always have at least two boundary nodes')
        b_nodes = self.zx_diagram.b_nodes()
        for c in nx.connected_components(self.zx_diagram.copy()):
            if b_nodes.isdisjoint(c):
                self.zx_diagram.remove_nodes_from(c)

    def __is_simplified(self) -> bool:
        num_non_zero_phases = 0
        for n, phase in self.zx_diagram.phases():
            if phase != 0.:
                num_non_zero_phases += 1
        return num_non_zero_phases == 0

    def reset(self) -> Any:
        self.zx_diagram = clifford_zx_diagram(self.num_qubits, self.depth, self.t_gates)
        self.zx_match_diagram = to_zx_match_diagram(self.zx_diagram)
        self.previous_value = diagram_value(self.zx_diagram)
        self.episode_return = 0
        self.episode_length = 0
        self.done = False
        self.previous_reward = 0
        return {
            'observation': self.zx_match_diagram.to_pyg_hdata(),
            'reward': self.previous_reward,
            'done': self.done,
        }

    def close(self) -> None:
        pass

    def __diagram_value(self) -> int:
        """
        TODO - Maybe not this simple...
        -1 for every node
        -1 for every edge
        -1 for every non-Clifford gate.
        """
        return -self.zx_diagram.number_of_nodes() - sum(
            [1 if p % 0.5 != 0 else 0 for p in self.zx_diagram.phases().values()]) - len(self.zx_diagram.edges())

    def __action_to_match(self, action: tuple) -> Tuple[Match, Optional[FRightParameters]]:
        action_type = action[0]
        node = action[1]
        if self.hdata_node_index is not None:
            match = self.hdata_node_index[(action_type, node)]
        else:
            raise Exception("Expected 'hdata_node_index' to be defined")
        if action_type == FRightZMatch.index:
            assert_correct_match_instance(FRightZMatch, match)
            phase = cat_phase_to_float(action[2], self.zx_match_diagram.phase_denominator)
            new_edges = cat_new_edges_to_int(action[3])
            transfer_edges = bernoulli_transfer_edges_to_set(self.zx_match_diagram, action[4:])
            return match, FRightParameters(phase, new_edges, transfer_edges)
        elif action_type == FLeftZMatch.index:
            assert_correct_match_instance(FLeftZMatch, match)
            raise Exception('Not implemented')
        elif action_type == FRightXMatch.index:
            assert_correct_match_instance(FRightXMatch, match)
            raise Exception('Not implemented')
        elif action_type == FLeftXMatch.index:
            assert_correct_match_instance(FLeftXMatch, match)
            raise Exception('Not implemented')
        elif action_type == BRightMatch.index:
            assert_correct_match_instance(BRightMatch, match)
            raise Exception('Not implemented')
        elif action_type == BLeftMatch.index:
            assert_correct_match_instance(BLeftMatch, match)
            raise Exception('Not implemented')
        elif action_type == YRightZMatch.index:
            assert_correct_match_instance(YRightZMatch, match)
            raise Exception('Not implemented')
        elif action_type == YLeftZMatch.index:
            assert_correct_match_instance(YLeftZMatch, match)
            raise Exception('Not implemented')
        elif action_type == YRightXMatch.index:
            assert_correct_match_instance(YRightXMatch, match)
            raise Exception('Not implemented')
        elif action_type == YLeftXMatch.index:
            assert_correct_match_instance(YLeftXMatch, match)
            raise Exception('Not implemented')
        else:
            raise ValueError(f'Unexpected action type {action_type}')

    def step(self, action: tuple) -> 'BaseEnv.timestep':
        self.episode_length += 1
        match, params = self.__action_to_match(action)
        rewrite(self.zx_diagram, match, params)
        self.__remove_isolated_nodes()
        self.__remove_self_loop_edges()
        self.__remove_isolated_components()
        self.done = self.__is_simplified()
        current_value = self.__diagram_value()
        self.previous_reward = self.previous_value - current_value + (0 if self.done else -self.step_penalty)
        self.episode_return += self.previous_reward
        self.previous_value = current_value
        self.zx_match_diagram = to_zx_match_diagram(self.zx_diagram)
        hdata, hdata_node_index = self.zx_match_diagram.to_pyg_hdata(True)
        self.hdata_node_index = hdata_node_index
        return {
            'observation': hdata,
            'reward': self.previous_reward,
            'done': self.done,
        }

    def seed(self, seed: int) -> None:
        pass

    def __repr__(self) -> str:
        pass

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def get_done_reward(self) -> Tuple[bool, Optional[float]]:
        return self.done, self.done_reward if self.done is True else None
