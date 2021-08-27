import os
from typing import Dict, Any

from ray.rllib.env import MultiAgentEnv
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.utils import merge_dicts

from grl.rl_apps.scenarios.trainer_configs.defaults import GRL_DEFAULT_OSHI_ZUMO_MEDIUM_DQN_PARAMS
from grl.rl_apps.scenarios.trainer_configs.defaults import GRL_DEFAULT_POKER_PPO_PARAMS
from grl.rllib_tools.stochastic_sampling_ignore_kwargs import StochasticSamplingIgnoreKwargs
from grl.rllib_tools.valid_actions_epsilon_greedy import ValidActionsEpsilonGreedy


def attack_and_counter_psro_ppo_params(env: MultiAgentEnv) -> Dict[str, Any]:
    return merge_dicts(GRL_DEFAULT_POKER_PPO_PARAMS, {
        "num_gpus": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_workers": 4,
        "num_gpus_per_worker": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_envs_per_worker": 1,
        "metrics_smoothing_episodes": 10000,
        "exploration_config": {
            # The Exploration class to use. In the simplest case, this is the name
            # (str) of any class present in the `rllib.utils.exploration` package.
            # You can also provide the python class directly or the full location
            # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
            # EpsilonGreedy").
            "type": StochasticSamplingIgnoreKwargs,
            # Add constructor kwargs here (if any).
        },
        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_hiddens": [32, 32],
            "custom_action_dist": "TorchGaussianSquashedGaussian",
        }),
        "entropy_coeff": 0.0,
        "lambda": 0.9,
        "train_batch_size": 2048,
        "sgd_minibatch_size": 512,
        "num_sgd_iter": 30,
        "lr": 0.0005,
        "clip_param": 0.2,
        "kl_target": 0.1,
    })


def loss_game_psro_ppo_params(env: MultiAgentEnv) -> Dict[str, Any]:
    return merge_dicts(GRL_DEFAULT_POKER_PPO_PARAMS, {
        "num_gpus": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_workers": 4,
        "num_gpus_per_worker": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_envs_per_worker": 1,
        "metrics_smoothing_episodes": 10000,
        "exploration_config": {
            # The Exploration class to use. In the simplest case, this is the name
            # (str) of any class present in the `rllib.utils.exploration` package.
            # You can also provide the python class directly or the full location
            # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
            # EpsilonGreedy").
            "type": StochasticSamplingIgnoreKwargs,
            # Add constructor kwargs here (if any).
        },
        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_hiddens": [32, 32],
            "custom_action_dist": "TorchGaussianSquashedGaussian",
        }),
        "entropy_coeff": 0.01,
        "lambda": 1.0,
        "train_batch_size": 2048,
        "sgd_minibatch_size": 256,
        "num_sgd_iter": 30,
        "lr": 0.0005,
        "clip_param": 0.2,
        "kl_target": 0.01,
    })


def loss_game_psro_ppo_params_orig(env: MultiAgentEnv) -> Dict[str, Any]:
    return merge_dicts(GRL_DEFAULT_POKER_PPO_PARAMS, {
        "num_gpus": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_workers": 4,
        "num_gpus_per_worker": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_envs_per_worker": 1,
        "metrics_smoothing_episodes": 10000,
        "exploration_config": {
            # The Exploration class to use. In the simplest case, this is the name
            # (str) of any class present in the `rllib.utils.exploration` package.
            # You can also provide the python class directly or the full location
            # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
            # EpsilonGreedy").
            "type": StochasticSamplingIgnoreKwargs,
            # Add constructor kwargs here (if any).
        },
        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_hiddens": [32, 32],
            "custom_action_dist": "TorchGaussianSquashedGaussian",
        }),
    })


def loss_game_nfsp_dqn_params(env: MultiAgentEnv) -> Dict[str, Any]:
    return merge_dicts(GRL_DEFAULT_OSHI_ZUMO_MEDIUM_DQN_PARAMS, {
        "metrics_smoothing_episodes": 10000,

        "exploration_config": {
            "epsilon_timesteps": int(500e6),
            "final_epsilon": 0.001,
            "initial_epsilon": 0.06,
            "type": ValidActionsEpsilonGreedy
        },
        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_hiddens": [32, 32],
        }),
        "target_network_update_freq": 100000,
        "buffer_size": 100000,
        "lr": 0.007,
        "rollout_fragment_length": 16,
        "train_batch_size": 4096,
    })


def loss_game_nfsp_avg_policy_params(env: MultiAgentEnv) -> Dict[str, Any]:
    return {
        "metrics_smoothing_episodes": 10000,

        "framework": "torch",
        "num_gpus": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_workers": 0,
        "num_gpus_per_worker": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_envs_per_worker": 1,
        "learning_starts": 16000,
        "train_batch_size": 4096,
        "lr": 0.07,
        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_hiddens": [32, 32],
        }),
    }


def loss_game_nfsp_dqn_params_orig(env: MultiAgentEnv) -> Dict[str, Any]:
    return merge_dicts(GRL_DEFAULT_OSHI_ZUMO_MEDIUM_DQN_PARAMS, {
        "metrics_smoothing_episodes": 10000,

        "exploration_config": {
            "epsilon_timesteps": int(500e6),
            "final_epsilon": 0.001,
            "initial_epsilon": 0.06,
            "type": ValidActionsEpsilonGreedy
        },
        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_hiddens": [32, 32],
        }),
    })


def loss_game_nfsp_avg_policy_params_orig(env: MultiAgentEnv) -> Dict[str, Any]:
    return {
        "metrics_smoothing_episodes": 10000,

        "framework": "torch",
        "num_gpus": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_workers": 0,
        "num_gpus_per_worker": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_envs_per_worker": 1,
        "learning_starts": 16000,
        "train_batch_size": 2048,
        "lr": 0.1,
        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_hiddens": [32, 32],
        }),
    }
