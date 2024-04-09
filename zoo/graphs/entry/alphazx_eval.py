import numpy as np

from lzero.entry import eval_alphazx
from zoo.graphs.config.alphazx_config import main_config, create_config

if __name__ == '__main__':
    """ 
    model_path (:obj:`Optional[str]`): The pretrained model path, which should
    point to the ckpt file of the pretrained model, and an absolute path is recommended.
    In LightZero, the path is usually something like ``exp_name/ckpt/ckpt_best.pth.tar``.
    """
    # model_path = './ckpt/ckpt_best.pth.tar'
    model_path = None
    seeds = [0]
    num_episodes_each_seed = 1
    # If True, you can play with the agent.
    main_config.env.agent_vs_human = False
    main_config.env.battle_mode = 'eval_mode'
    main_config.env.render_mode = 'image_realtime_mode'
    main_config.max_num_qubits = 50
    main_config.max_circuit_depth = 50,
    main_config.t_gates = True,
    main_config.max_num_new_edges = 10,
    main_config.num_phase_buckets = 10,
    main_config.done_reward = 1.,
    main_config.step_penalty = -1.,
    create_config.env_manager.type = 'base'
    main_config.env.collector_env_num = 1
    main_config.env.evaluator_env_num = 1
    main_config.env.n_evaluator_episode = 1
    total_test_episodes = num_episodes_each_seed * len(seeds)
    returns_mean_seeds = []
    returns_seeds = []
    for seed in seeds:
        returns_mean, returns = eval_alphazx(
            (main_config, create_config),
            seed=seed,
            num_episodes_each_seed=num_episodes_each_seed,
            print_seed_details=True,
            model_path=model_path
        )
        returns_mean_seeds.append(returns_mean)
        returns_seeds.append(returns)

    returns_mean_seeds = np.array(returns_mean_seeds)
    returns_seeds = np.array(returns_seeds)

    print("=" * 20)
    print(f'We eval total {len(seeds)} seeds. In each seed, we eval {num_episodes_each_seed} episodes.')
    print(f'In seeds {seeds}, returns_mean_seeds is {returns_mean_seeds}, returns is {returns_seeds}')
    print('In all seeds, reward_mean:', returns_mean_seeds.mean(), end='. ')
    print(
        f'win rate: {len(np.where(returns_seeds == 1.)[0]) / total_test_episodes}, draw rate: {len(np.where(returns_seeds == 0.)[0]) / total_test_episodes}, lose rate: {len(np.where(returns_seeds == -1.)[0]) / total_test_episodes}'
    )
    print("=" * 20)
