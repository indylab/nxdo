import os
from typing import Dict, Any

from ray.rllib.env import MultiAgentEnv
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.utils import merge_dicts

from grl.rl_apps.scenarios.trainer_configs.defaults import GRL_DEFAULT_OPENSPIEL_POKER_DQN_PARAMS
from grl.rllib_tools.models.valid_actions_fcnet import get_valid_action_fcn_class_for_env
from grl.rllib_tools.valid_actions_epsilon_greedy import ValidActionsEpsilonGreedy


def nfsp_kuhn_avg_policy_params_openspiel(env: MultiAgentEnv) -> Dict[str, Any]:
    return {
        "framework": "torch",
        "learning_starts": 2000,
        "train_batch_size": 128,
        "lr": 0.01,
        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128],
        }),
    }


def nfsp_kuhn_dqn_params_openspiel(env: MultiAgentEnv) -> Dict[str, Any]:
    return GRL_DEFAULT_OPENSPIEL_POKER_DQN_PARAMS


def nfsp_leduc_avg_policy_params_openspiel(env: MultiAgentEnv) -> Dict[str, Any]:
    return {
        "framework": "torch",
        "learning_starts": 2000,
        "train_batch_size": 128,
        "lr": 0.01,
        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128],
            "custom_model": get_valid_action_fcn_class_for_env(env=env)
        }),
    }


def nfsp_leduc_dqn_params_openspeil(env: MultiAgentEnv) -> Dict[str, Any]:
    return merge_dicts(GRL_DEFAULT_OPENSPIEL_POKER_DQN_PARAMS, {
        # === Exploration Settings ===
        "exploration_config": {
            # The Exploration class to use.
            "type": ValidActionsEpsilonGreedy,
            # Config for the Exploration class' constructor:
            "initial_epsilon": 0.06,
            "final_epsilon": 0.001,
            "epsilon_timesteps": int(20e6) * 10,  # Timesteps over which to anneal epsilon.
        },
        # Update the target network every `target_network_update_freq` steps.
        "target_network_update_freq": 19200 * 10,

        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128],
            "custom_model": get_valid_action_fcn_class_for_env(env=env),
        }),
    })


def nfsp_kuhn_avg_policy_params(env: MultiAgentEnv) -> Dict[str, Any]:
    return {
        "framework": "torch",
        "num_gpus": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_workers": 0,
        "num_gpus_per_worker": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_envs_per_worker": 1,
        "learning_starts": 16000,
        "train_batch_size": 4096,
        "lr": 0.1,
        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128],
        }),
    }


def nfsp_kuhn_dqn_params(env: MultiAgentEnv) -> Dict[str, Any]:
    return merge_dicts(GRL_DEFAULT_OPENSPIEL_POKER_DQN_PARAMS, {
        "num_gpus": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_workers": 4,
        "num_gpus_per_worker": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_envs_per_worker": 32,

        # How many steps of the model to sample before learning starts.
        "learning_starts": 16000,
        # Update the replay buffer with this many samples at once. Note that
        # this setting applies per-worker if num_workers > 1.q
        "rollout_fragment_length": 8,

        "batch_mode": "truncate_episodes",

        # Size of a batch sampled from replay buffer for training. Note that
        # if async_updates is set, then each worker returns gradients for a
        # batch of this size.
        "train_batch_size": 4096,
    })


def nfsp_leduc_avg_policy_params(env: MultiAgentEnv) -> Dict[str, Any]:
    return {
        "framework": "torch",
        "num_gpus": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_workers": 0,
        "num_gpus_per_worker": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_envs_per_worker": 1,
        "learning_starts": 16000,
        "train_batch_size": 4096,
        "lr": 0.1,
        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128],
            "custom_model": get_valid_action_fcn_class_for_env(env=env),
        }),
    }


def nfsp_leduc_avg_policy_params_improved(env: MultiAgentEnv) -> Dict[str, Any]:
    # 09.23.29PM_Apr-30-2021/ orig dqn ant_prm FIXED 0.1 lr (0.3, 0.1) annealed 50000000 steps/leduc_nfsp_dqn_hparam_search_nfsp_sparse_09.24.43PM_Apr-30-2021l2zf5w7z

    return {
        "framework": "torch",
        "num_gpus": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_workers": 0,
        "num_gpus_per_worker": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_envs_per_worker": 1,
        "learning_starts": 16000,
        "train_batch_size": 4096,
        # "lr": avg_pol_lr_start_end[0],
        "lr_schedule": [[0, 0.3], [50000000, 0.1]],
        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128],
            "custom_model": get_valid_action_fcn_class_for_env(env=env),
        }),
    }


def nfsp_leduc_dqn_params(env: MultiAgentEnv) -> Dict[str, Any]:
    return merge_dicts(GRL_DEFAULT_OPENSPIEL_POKER_DQN_PARAMS, {
        # === Exploration Settings ===
        "exploration_config": {
            # The Exploration class to use.
            "type": ValidActionsEpsilonGreedy,
            # Config for the Exploration class' constructor:
            "initial_epsilon": 0.06,
            "final_epsilon": 0.001,
            "epsilon_timesteps": int(20e6),  # Timesteps over which to anneal epsilon.
        },

        "num_gpus": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_workers": 4,
        "num_gpus_per_worker": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_envs_per_worker": 32,

        # How many steps of the model to sample before learning starts.
        "learning_starts": 16000,
        # Update the replay buffer with this many samples at once. Note that
        # this setting applies per-worker if num_workers > 1.q
        "rollout_fragment_length": 8,

        # Size of a batch sampled from replay buffer for training. Note that
        # if async_updates is set, then each worker returns gradients for a
        # batch of this size.
        "train_batch_size": 4096,

        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128],
            "custom_model": get_valid_action_fcn_class_for_env(env=env),
        }),
    })


def nfsp_leduc_dqn_params_two_layers_no_valid_actions_model(env: MultiAgentEnv) -> Dict[str, Any]:
    params = nfsp_leduc_dqn_params(env=env)
    params["model"]["custom_model"] = None
    params["model"]["fcnet_hiddens"] = [128, 128]
    return params


def nfsp_leduc_avg_policy_params_two_layers_no_valid_actions_model(env: MultiAgentEnv) -> Dict[str, Any]:
    params = nfsp_leduc_avg_policy_params(env=env)
    params["model"]["custom_model"] = None
    params["model"]["fcnet_hiddens"] = [128, 128]
    return params


def nfsp_oshi_ppo_params_two_layers_no_valid_actions_model(env: MultiAgentEnv) -> Dict[str, Any]:
    params = nfsp_leduc_dqn_params(env=env)
    params["model"]["custom_model"] = None
    params["model"]["fcnet_hiddens"] = [64, 64]
    return params


def nfsp_oshi_ppo_avg_policy_params_two_layers_no_valid_actions_model(env: MultiAgentEnv) -> Dict[str, Any]:
    params = nfsp_leduc_avg_policy_params(env=env)
    params["model"]["custom_model"] = None
    params["model"]["fcnet_hiddens"] = [64, 64]
    return params


def larger_nfsp_oshi_ppo_params_two_layers_no_valid_actions_model(env: MultiAgentEnv) -> Dict[str, Any]:
    params = nfsp_leduc_dqn_params(env=env)
    params["metrics_smoothing_episodes"] = 5000
    params["model"]["custom_model"] = None
    params["model"]["fcnet_hiddens"] = [128, 128]
    return params


def larger_nfsp_oshi_ppo_avg_policy_params_two_layers_no_valid_actions_model(env: MultiAgentEnv) -> Dict[str, Any]:
    params = nfsp_leduc_avg_policy_params(env=env)
    params["metrics_smoothing_episodes"] = 5000
    params["model"]["custom_model"] = None
    params["model"]["fcnet_hiddens"] = [128, 128]
    return params


def larger_nfsp_oshi_ppo_params_two_layers_with_valid_actions_model(env: MultiAgentEnv) -> Dict[str, Any]:
    params = nfsp_leduc_dqn_params(env=env)
    params["metrics_smoothing_episodes"] = 5000
    params["model"]["fcnet_hiddens"] = [128, 128]
    return params


def larger_nfsp_oshi_ppo_avg_policy_params_two_layers_with_valid_actions_model(env: MultiAgentEnv) -> Dict[str, Any]:
    params = nfsp_leduc_avg_policy_params(env=env)
    params["metrics_smoothing_episodes"] = 5000
    params["model"]["fcnet_hiddens"] = [128, 128]
    return params
