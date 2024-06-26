from easydict import EasyDict

from . import gomoku_play_with_bot
from . import gym_breakoutnoframeskip_v4
from . import gym_cartpole_v0
from . import gym_lunarlander_v2
from . import gym_mspacmannoframeskip_v4
from . import gym_pendulum_v1
from . import gym_pongnoframeskip_v4
from . import tictactoe_play_with_bot

supported_env_cfg = {
    gomoku_play_with_bot.cfg.main_config.env.env_id: gomoku_play_with_bot.cfg,
    gym_breakoutnoframeskip_v4.cfg.main_config.env.env_id: gym_breakoutnoframeskip_v4.cfg,
    gym_cartpole_v0.cfg.main_config.env.env_id: gym_cartpole_v0.cfg,
    gym_lunarlander_v2.cfg.main_config.env.env_id: gym_lunarlander_v2.cfg,
    gym_mspacmannoframeskip_v4.cfg.main_config.env.env_id: gym_mspacmannoframeskip_v4.cfg,
    gym_pendulum_v1.cfg.main_config.env.env_id: gym_pendulum_v1.cfg,
    gym_pongnoframeskip_v4.cfg.main_config.env.env_id: gym_pongnoframeskip_v4.cfg,
    tictactoe_play_with_bot.cfg.main_config.env.env_id: tictactoe_play_with_bot.cfg,
}

supported_env_cfg = EasyDict(supported_env_cfg)
