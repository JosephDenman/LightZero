from easydict import EasyDict

from lzero.mcts.tree_search import mcts_ctree

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 9
n_episode = 8
evaluator_env_num = 6
num_simulations = 25
update_per_collect = 50
batch_size = 256
max_env_step = int(2e5)
num_of_sampled_actions = 50
mcts_ctree = False
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

cfg = dict(
    main_config=dict(
        exp_name='alphazx',
        seed=0,
        env=dict(
            env_id='alphazx',
            battle_mode='self_play',
            max_num_qubits=50,
            max_circuit_depth=50,
            t_gates=True,
            max_num_new_edges=10,
            num_phase_buckets=10,
            done_reward=1.,
            step_penalty=-1.,
            bot_action_type='v0',  # {'v0', 'alpha_beta_pruning'}
            channel_last=False,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, ),
            # ==============================================================
            # for the creation of simulation env
            agent_vs_human=False,
            prob_random_agent=0,
            prob_expert_agent=0,
            scale=True,
            alphazero_mcts_ctree=mcts_ctree,
            save_replay_gif=False,
            replay_path_gif='./replay_gif',
            # ==============================================================
        ),
        policy=dict(
            # ==============================================================
            # for the creation of simulation env
            simulation_env_id='alphazx',
            simulation_env_config_type='self_play',
            # ==============================================================
            model=dict(
                observation_shape='hdata',
                action_space_size=5
            ),
            sampled_algo=True,
            mcts_ctree=mcts_ctree,
            policy_loss_type='cross_entropy',
            cuda=True,
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            optim_type='Adam',
            lr_piecewise_constant_decay=False,
            learning_rate=0.003,
            grad_clip_value=0.5,
            value_weight=1.0,
            entropy_weight=0.0,
            n_episode=n_episode,
            eval_freq=int(2e3),
            mcts=dict(num_simulations=num_simulations, num_of_sampled_actions=num_of_sampled_actions),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
        ),
        wandb_logger=dict(
            gradient_logger=False, video_logger=False, plot_logger=False, action_logger=False, return_logger=False
        ),
    ),
    create_config=dict(
        env=dict(
            type='alphazx',
            import_names=['zoo.graphs.envs.alphazx_env'],
        ),
        env_manager=dict(type='subprocess'),
        policy=dict(
            type='alphazx',
            import_names=['lzero.policy.alphazx'],
        ),
        collector=dict(
            type='episode_alphazero',
            import_names=['lzero.worker.alphazx_collector'],
        ),
        evaluator=dict(
            type='alphazero',
            import_names=['lzero.worker.alphazx_evaluator'],
        ),
    )
)

cfg = EasyDict(cfg)
