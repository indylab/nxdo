import os
from typing import Dict, Any

from ray.rllib.env import MultiAgentEnv
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.utils import merge_dicts

from grl.rl_apps.scenarios.trainer_configs.defaults import GRL_DEFAULT_OPENSPIEL_POKER_DQN_PARAMS, \
    GRL_DEFAULT_POKER_PPO_PARAMS
from grl.rllib_tools.valid_actions_epsilon_greedy import ValidActionsEpsilonGreedy
from grl.rllib_tools.valid_actions_fcnet import get_valid_action_fcn_class_for_env


def psro_kuhn_dqn_params_openspiel(env: MultiAgentEnv) -> Dict[str, Any]:
    return GRL_DEFAULT_OPENSPIEL_POKER_DQN_PARAMS


def psro_leduc_dqn_params_openspiel(env: MultiAgentEnv) -> Dict[str, Any]:
    return merge_dicts(GRL_DEFAULT_OPENSPIEL_POKER_DQN_PARAMS, {
        # === Exploration Settings ===
        "exploration_config": {
            # The Exploration class to use.
            "type": ValidActionsEpsilonGreedy,
            # Config for the Exploration class constructor:
            "initial_epsilon": 0.06,
            "final_epsilon": 0.001,
            "epsilon_timesteps": int(20e6) * 10,  # Timesteps over which to anneal epsilon.

        },

        # Update the target network every `target_network_update_freq` steps.
        "target_network_update_freq": 19200 * 10,

        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128],
            "custom_model": get_valid_action_fcn_class_for_env(env=env)
        }),
    })


def psro_leduc_ppo_params(env: MultiAgentEnv) -> Dict[str, Any]:
    return merge_dicts(GRL_DEFAULT_POKER_PPO_PARAMS, {
        "num_gpus": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_workers": 4,
        "num_gpus_per_worker": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_envs_per_worker": 1,
    })


def psro_leduc_dqn_params(env: MultiAgentEnv) -> Dict[str, Any]:
    return merge_dicts(GRL_DEFAULT_OPENSPIEL_POKER_DQN_PARAMS, {
        # === Exploration Settings ===
        "exploration_config": {
            # The Exploration class to use.
            "type": ValidActionsEpsilonGreedy,
            # Config for the Exploration class constructor:
            "initial_epsilon": 0.20,
            "final_epsilon": 0.001,
            "epsilon_timesteps": int(1e5),  # Timesteps over which to anneal epsilon.
        },

        # Update the target network every `target_network_update_freq` steps.
        "target_network_update_freq": 10000,

        "num_gpus": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_workers": 4,
        "num_gpus_per_worker": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_envs_per_worker": 1,

        # How many steps of the model to sample before learning starts.
        "learning_starts": 16000,
        # Update the replay buffer with this many samples at once. Note that
        # this setting applies per-worker if num_workers > 1.q
        "rollout_fragment_length": 8 * 32,

        # Size of a batch sampled from replay buffer for training. Note that
        # if async_updates is set, then each worker returns gradients for a
        # batch of this size.
        "train_batch_size": 4096,

        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128],
            "custom_model": get_valid_action_fcn_class_for_env(env=env)
        }),
    })


def psro_kuhn_dqn_params(env: MultiAgentEnv) -> Dict[str, Any]:
    return merge_dicts(GRL_DEFAULT_OPENSPIEL_POKER_DQN_PARAMS, {
        # === Exploration Settings ===
        "exploration_config": {
            # The Exploration class to use.
            "type": ValidActionsEpsilonGreedy,
            # Config for the Exploration class constructor:
            "initial_epsilon": 0.20,
            "final_epsilon": 0.001,
            "epsilon_timesteps": int(1e5),  # Timesteps over which to anneal epsilon.
        },

        "num_gpus": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_workers": 4,
        "num_gpus_per_worker": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_envs_per_worker": 1,

        # How many steps of the model to sample before learning starts.
        "learning_starts": 16000,
        # Update the replay buffer with this many samples at once. Note that
        # this setting applies per-worker if num_workers > 1.q
        "rollout_fragment_length": 8 * 32,

        # Size of a batch sampled from replay buffer for training. Note that
        # if async_updates is set, then each worker returns gradients for a
        # batch of this size.
        "train_batch_size": 4096,
    })
