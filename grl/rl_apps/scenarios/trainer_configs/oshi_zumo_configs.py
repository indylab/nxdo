import os
from typing import Dict, Any

from ray.rllib.env import MultiAgentEnv
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.utils import merge_dicts

from grl.rl_apps.scenarios.trainer_configs.defaults import GRL_DEFAULT_OSHI_ZUMO_MEDIUM_DQN_PARAMS, \
    GRL_DEFAULT_OSHI_ZUMO_TINY_DQN_PARAMS
from grl.rllib_tools.models.valid_actions_fcnet import get_valid_action_fcn_class_for_env
from grl.rllib_tools.valid_actions_epsilon_greedy import ValidActionsEpsilonGreedy


def medium_oshi_zumo_psro_dqn_params(env: MultiAgentEnv) -> Dict[str, Any]:
    return merge_dicts(GRL_DEFAULT_OSHI_ZUMO_MEDIUM_DQN_PARAMS, {
        "exploration_config": {
            "epsilon_timesteps": int(4e6),
            "final_epsilon": 0.001,
            "initial_epsilon": 0.06,
            "type": ValidActionsEpsilonGreedy
        },
        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128, 128],
            "custom_model": get_valid_action_fcn_class_for_env(env=env),
        }),
    })


def medium_oshi_zumo_nfsp_dqn_params(env: MultiAgentEnv) -> Dict[str, Any]:
    return merge_dicts(GRL_DEFAULT_OSHI_ZUMO_MEDIUM_DQN_PARAMS, {
        "exploration_config": {
            "epsilon_timesteps": int(500e6),
            "final_epsilon": 0.001,
            "initial_epsilon": 0.06,
            "type": ValidActionsEpsilonGreedy
        },
        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128, 128],
            "custom_model": get_valid_action_fcn_class_for_env(env=env),
        }),
    })


def medium_oshi_zumo_nfsp_avg_policy_params(env: MultiAgentEnv) -> Dict[str, Any]:
    return {
        "framework": "torch",
        "num_gpus": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_workers": 0,
        "num_gpus_per_worker": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_envs_per_worker": 1,
        "learning_starts": 16000,
        "train_batch_size": 2048,
        "lr": 0.1,
        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128, 128],
            "custom_model": get_valid_action_fcn_class_for_env(env=env),
        }),
    }


def tiny_oshi_zumo_psro_dqn_params(env: MultiAgentEnv) -> Dict[str, Any]:
    return merge_dicts(GRL_DEFAULT_OSHI_ZUMO_TINY_DQN_PARAMS, {
        "exploration_config": {
            "epsilon_timesteps": int(400e3),
            "final_epsilon": 0.001,
            "initial_epsilon": 0.06,
            "type": ValidActionsEpsilonGreedy
        },
        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128, 128],
            "custom_model": get_valid_action_fcn_class_for_env(env=env),
        }),
    })


def tiny_oshi_zumo_nfsp_dqn_params(env: MultiAgentEnv) -> Dict[str, Any]:
    return merge_dicts(GRL_DEFAULT_OSHI_ZUMO_TINY_DQN_PARAMS, {
        "exploration_config": {
            "epsilon_timesteps": int(100e6),
            "final_epsilon": 0.001,
            "initial_epsilon": 0.06,
            "type": ValidActionsEpsilonGreedy
        },
        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128, 128],
            "custom_model": get_valid_action_fcn_class_for_env(env=env),
        }),
    })


def tiny_oshi_zumo_nfsp_avg_policy_params(env: MultiAgentEnv) -> Dict[str, Any]:
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
            "fcnet_hiddens": [128, 128],
            "custom_model": get_valid_action_fcn_class_for_env(env=env),
        }),
    }
