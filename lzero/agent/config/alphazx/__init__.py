from easydict import EasyDict

from . import alphazx

supported_env_cfg = {
    alphazx.cfg.main_config.env.env_id: alphazx.cfg
}

supported_env_cfg = EasyDict(supported_env_cfg)
