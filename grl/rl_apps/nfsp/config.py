import numpy as np
from typing import Dict
from gym.spaces import Space, Discrete, Box

from ray.rllib.utils import merge_dicts
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch


# def debug_before_learn_on_batch(x: MultiAgentBatch, *args, **kwargs):
#     print(f"DQN trainer learn on batch {[s.count for s in x.policy_batches.values()]}")
#     return x

def kuhn_dqn_params(action_space: Space) -> Dict:
    return {
        "framework": "torch",
        # Smooth metrics over this many episodes.
        "metrics_smoothing_episodes": 1000,
        # === Model ===
        # Number of atoms for representing the distribution of return. When
        # this is greater than 1, distributional Q-learning is used.
        # the discrete supports are bounded by v_min and v_max
        "num_atoms": 1,
        "v_min": -10.0,
        "v_max": 10.0,
        # Whether to use noisy network
        "noisy": False,
        # control the initial value of noisy nets
        "sigma0": 0.5,
        # Whether to use dueling dqn
        "dueling": False,
        # Dense-layer setup for each the advantage branch and the value branch
        # in a dueling architecture.
        "hiddens": [256],
        # Whether to use double dqn
        # "double_q": False,
        "double_q": True,

        # N-step Q learning
        "n_step": 1,

        # === Exploration Settings ===
        "exploration_config": {
            # The Exploration class to use.
            "type": "EpsilonGreedy",
            # Config for the Exploration class' constructor:
            "initial_epsilon": 0.06,
            "final_epsilon": 0.001,
            "epsilon_timesteps": int(3e7),  # Timesteps over which to anneal epsilon.

            # For soft_q, use:
            # "exploration_config" = {
            #   "type": "SoftQ"
            #   "temperature": [float, e.g. 1.0]
            # }
        },
        # Switch to greedy actions in evaluation workers.
        "evaluation_config": {
            "explore": False,
        },
        "explore": True,

        # Update the target network every `target_network_update_freq` steps.
        "target_network_update_freq": 10000,
        # "target_network_update_freq": 1,

        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": int(2e5),

        # If True prioritized replay buffer will be used.
        "prioritized_replay": False,
        # Alpha parameter for prioritized replay buffer.
        "prioritized_replay_alpha": 0.0,
        # Beta parameter for sampling from prioritized replay buffer.
        "prioritized_replay_beta": 0.0,
        # Final value of beta (by default, we use constant beta=0.4).
        "final_prioritized_replay_beta": 0.0,
        # Time steps over which the beta parameter is annealed.
        "prioritized_replay_beta_annealing_timesteps": 20000,
        # Epsilon to add to the TD errors when updating priorities.
        "prioritized_replay_eps": 0.0,
        # Whether to LZ4 compress observations
        "compress_observations": False,
        # Callback to run before learning on a multi-agent batch of experiences.
        # "before_learn_on_batch": debug_before_learn_on_batch,
        # If set, this will fix the ratio of replayed from a buffer and learned on
        # timesteps to sampled from an environment and stored in the replay buffer
        # timesteps. Otherwise, the replay will proceed at the native ratio
        # determined by (train_batch_size / rollout_fragment_length).
        "training_intensity": None,

        # === Optimization ===
        # Learning rate for adam optimizer
        "lr": 0.01,
        # Learning rate schedule
        "lr_schedule": None,
        # Adam epsilon hyper parameter
        "adam_epsilon": 1e-8,
        # If not None, clip gradients during optimization at this value
        "grad_clip": None,
        # How many steps of the model to sample before learning starts.
        "learning_starts": 2000,
        # Update the replay buffer with this many samples at once. Note that
        # this setting applies per-worker if num_workers > 1.q
        "rollout_fragment_length": 64,
        "batch_mode": "truncate_episodes",

        # "rollout_fragment_length": 1,
        # "batch_mode": "complete_episodes",

        # Size of a batch sampled from replay buffer for training. Note that
        # if async_updates is set, then each worker returns gradients for a
        # batch of this size.
        "train_batch_size": 128,

        # Whether to compute priorities on workers.
        "worker_side_prioritization": False,
        # Prevent iterations from going lower than this time span
        "min_iter_time_s": 0,
        # Minimum env steps to optimize for per train call. This value does
        # not affect learning (JB: this is a lie!), only the length of train iterations.
        "timesteps_per_iteration": 0,


        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128],
        }),
    }


def leduc_dqn_params(action_space: Space) -> Dict:
    return {
        "framework": "torch",
        # Smooth metrics over this many episodes.
        "metrics_smoothing_episodes": 1000,
        # === Model ===
        # Number of atoms for representing the distribution of return. When
        # this is greater than 1, distributional Q-learning is used.
        # the discrete supports are bounded by v_min and v_max
        "num_atoms": 1,
        "v_min": -10.0,
        "v_max": 10.0,
        # Whether to use noisy network
        "noisy": False,
        # control the initial value of noisy nets
        "sigma0": 0.5,
        # Whether to use dueling dqn
        "dueling": False,
        # Dense-layer setup for each the advantage branch and the value branch
        # in a dueling architecture.
        "hiddens": [256],
        # Whether to use double dqn
        "double_q": True,
        # N-step Q learning
        "n_step": 1,

        # === Exploration Settings ===
        "exploration_config": {
            # The Exploration class to use.
            "type": "EpsilonGreedy",
            # Config for the Exploration class' constructor:
            "initial_epsilon": 0.06,
            "final_epsilon": 0.001,
            "epsilon_timesteps": int(20e6) * 10,  # Timesteps over which to anneal epsilon.

            # For soft_q, use:
            # "exploration_config" = {
            #   "type": "SoftQ"
            #   "temperature": [float, e.g. 1.0]
            # }
        },
        # Switch to greedy actions in evaluation workers.
        "evaluation_config": {
            "explore": False,
        },
        "explore": True,

        # Update the target network every `target_network_update_freq` steps.
        "target_network_update_freq": 19200 * 10,
        # "target_network_update_freq": 1,

        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": int(2e5),

        # If True prioritized replay buffer will be used.
        "prioritized_replay": False,
        # Alpha parameter for prioritized replay buffer.
        "prioritized_replay_alpha": 0.0,
        # Beta parameter for sampling from prioritized replay buffer.
        "prioritized_replay_beta": 0.0,
        # Final value of beta (by default, we use constant beta=0.4).
        "final_prioritized_replay_beta": 0.0,
        # Time steps over which the beta parameter is annealed.
        "prioritized_replay_beta_annealing_timesteps": 20000,
        # Epsilon to add to the TD errors when updating priorities.
        "prioritized_replay_eps": 0.0,
        # Whether to LZ4 compress observations
        "compress_observations": False,
        # Callback to run before learning on a multi-agent batch of experiences.
        # "before_learn_on_batch": debug_before_learn_on_batch,
        # If set, this will fix the ratio of replayed from a buffer and learned on
        # timesteps to sampled from an environment and stored in the replay buffer
        # timesteps. Otherwise, the replay will proceed at the native ratio
        # determined by (train_batch_size / rollout_fragment_length).
        "training_intensity": None,

        # === Optimization ===
        # Learning rate for adam optimizer
        "lr": 0.01,
        # Learning rate schedule
        "lr_schedule": None,
        # Adam epsilon hyper parameter
        "adam_epsilon": 1e-8,
        # If not None, clip gradients during optimization at this value
        "grad_clip": None,
        # How many steps of the model to sample before learning starts.
        "learning_starts": 2000,
        # Update the replay buffer with this many samples at once. Note that
        # this setting applies per-worker if num_workers > 1.q
        "rollout_fragment_length": 64,
        "batch_mode": "truncate_episodes",

        # "rollout_fragment_length": 1,
        # "batch_mode": "complete_episodes",

        # Size of a batch sampled from replay buffer for training. Note that
        # if async_updates is set, then each worker returns gradients for a
        # batch of this size.
        "train_batch_size": 128,

        # Whether to compute priorities on workers.
        "worker_side_prioritization": False,
        # Prevent iterations from going lower than this time span
        "min_iter_time_s": 0,
        # Minimum env steps to optimize for per train call. This value does
        # not affect learning (JB: this is a lie!), only the length of train iterations.
        "timesteps_per_iteration": 0,


        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128],
        }),
    }


def kuhn_sac_params(action_space: Space) -> Dict:
    if isinstance(action_space, Discrete):
        default_target_entropy = np.array(-np.log(1.0 / action_space.n), dtype=np.float32)
    else:
        default_target_entropy = -np.prod(action_space.shape)

    return {
        # Smooth metrics over this many episodes.
        "metrics_smoothing_episodes": 10000,

        "buffer_size": int(2e5),
        "learning_starts": 1000,
        # Update the target by \tau * policy + (1-\tau) * target_policy.
        # "tau": 1.0/10000,
        "tau": 1.0 / 150.0,

        "framework": "torch",

        "initial_alpha": 0.01,
        "target_entropy": 0.01 * default_target_entropy,
        "train_batch_size": 128,
        "rollout_fragment_length": 64,
        "normalize_actions": False,
        "model": merge_dicts(MODEL_DEFAULTS, {}),

        "Q_model": {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128],
        },
        # Model options for the policy function.
        "policy_model": {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128],
        },

        "use_state_preprocessor": False,
        "optimization": {
            "actor_learning_rate": 0.01,
            "critic_learning_rate": 0.01,
            "entropy_learning_rate": 0.01,
        },

        # Prevent iterations from going lower than this time span
        "min_iter_time_s": 0,
        # Minimum env steps to optimize for per train call. This value does
        # not affect learning (JB: this is a lie!), only the length of train iterations.
        "timesteps_per_iteration": 0,

    }