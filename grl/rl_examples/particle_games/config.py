import numpy as np
from gym.spaces import Box, Discrete, Space

from ray.rllib.utils import merge_dicts
from ray.rllib.models import MODEL_DEFAULTS


def simple_push_sac_params_small(action_space: Space):
    if isinstance(action_space, Discrete):
        default_target_entropy = np.array(-np.log(1.0 / action_space.n), dtype=np.float32)
    else:
        default_target_entropy = -np.prod(action_space.shape)

    return {
        # Smooth metrics over this many episodes.
        "metrics_smoothing_episodes": 3000,
        "framework": "torch",
        # RL Algo Specific

        "initial_alpha": 0.1,
        "target_entropy": 0.1 * default_target_entropy,
        "Q_model": {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128, 128],
        },
        # Model options for the policy function.
        "policy_model": {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128, 128],
        },
        "optimization": {
            "actor_learning_rate": 0.003,
            "critic_learning_rate": 0.003,
            "entropy_learning_rate": 0.003,
        },
        "normalize_actions": False,
        "train_batch_size": 512,
        "rollout_fragment_length": 8,
        "model": merge_dicts(MODEL_DEFAULTS, {}),
    }


def simple_push_sac_params(action_space: Space):
    if isinstance(action_space, Discrete):
        default_target_entropy = np.array(-np.log(1.0 / action_space.n), dtype=np.float32)
    else:
        default_target_entropy = -np.prod(action_space.shape)

    return {
        # Smooth metrics over this many episodes.
        "metrics_smoothing_episodes": 3000,
        "framework": "torch",
        # RL Algo Specific

        "initial_alpha": 0.1,
        "target_entropy": 0.1 * default_target_entropy,
        "Q_model": {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128, 128, 128],
        },
        # Model options for the policy function.
        "policy_model": {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128, 128, 128],
        },
        "optimization": {
            "actor_learning_rate": 3e-4,
            "critic_learning_rate": 3e-4,
            "entropy_learning_rate": 3e-4,
        },
        "normalize_actions": False,
        "train_batch_size": 1024,
        "rollout_fragment_length": 16,
        "model": merge_dicts(MODEL_DEFAULTS, {}),
    }


# def sumo_sac_params(action_space: Space):
#     if isinstance(action_space, Discrete):
#         default_target_entropy = np.array(-np.log(1.0 / action_space.n), dtype=np.float32)
#     else:
#         default_target_entropy = -np.prod(action_space.shape)
#
#     return {
#         # Smooth metrics over this many episodes.
#         "metrics_smoothing_episodes": 1000,
#         "framework": "torch",
#         # RL Algo Specific
#
#         "initial_alpha": 0.01,
#         "target_entropy": 0.3 * default_target_entropy,
#         "Q_model": {
#             "fcnet_activation": "relu",
#             "fcnet_hiddens": [128, 128],
#         },
#         # Model options for the policy function.
#         "policy_model": {
#             "fcnet_activation": "relu",
#             "fcnet_hiddens": [128, 128],
#         },
#         "optimization": {
#             "actor_learning_rate": 0.01,
#             "critic_learning_rate": 0.01,
#             "entropy_learning_rate": 0.01,
#         },
#         "tau": 0.01,
#         "normalize_actions": False,
#         "train_batch_size": 1024,
#         "rollout_fragment_length": 128,
#         "model": merge_dicts(MODEL_DEFAULTS, {}),
#     }


def sumo_ppo_params(action_space: Box):
    return {
        # Smooth metrics over this many episodes.
        "metrics_smoothing_episodes": 1000,
        "framework": "torch",

        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_hiddens": [64],
            # Nonlinearity for fully connected net (tanh, relu)
            "fcnet_activation": "tanh",
            # Filter config. List of [out_channels, kernel, stride] for each filter
            "conv_filters": None,
            # Nonlinearity for built-in convnet
            "conv_activation": "relu",
            # For DiagGaussian action distributions, make the second half of the model
            # outputs floating bias variables instead of state-dependent. This only
            # has an effect is using the default fully connected net.
            "free_log_std": False,
            # Whether to skip the final linear layer used to resize the hidden layer
            # outputs to size `num_outputs`. If True, then the last hidden layer
            # should already match num_outputs.
            "no_final_linear": False,
            # Whether layers should be shared for the value function.
            "vf_share_layers": False,

            # == LSTM ==
            # Whether to wrap the model with an LSTM.
            "use_lstm": True,
            # Max seq len for training the LSTM, defaults to 20.
            "max_seq_len": 10,
            # Size of the LSTM cell.
            "lstm_cell_size": 64,
            # Whether to feed a_{t-1}, r_{t-1} to LSTM.
            "lstm_use_prev_action_reward": True,
        }),

        # RL Algo Specific

        # Should use a critic as a baseline (otherwise don't use value baseline;
        # required for using GAE).
        "use_critic": True,
        # If true, use the Generalized Advantage Estimator (GAE)
        # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
        "use_gae": True,
        # The GAE(lambda) parameter.
        "lambda": 0.95,
        # Initial coefficient for KL divergence.
        "kl_coeff": 0.0,
        # Size of batches collected from each worker.
        "rollout_fragment_length": 200,
        # Number of timesteps collected for each SGD round. This defines the size
        # of each SGD epoch.
        "train_batch_size": 409600,
        # Total SGD batch size across all devices for SGD. This defines the
        # minibatch size within each epoch.
        "sgd_minibatch_size": 5120,
        # Whether to shuffle sequences in the batch when training (recommended).
        "shuffle_sequences": True,
        # Number of SGD iterations in each outer loop (i.e., number of epochs to
        # execute per train batch).
        "num_sgd_iter": 6,
        # Stepsize of SGD.
        "lr": 0.0003,
        # Learning rate schedule.
        "lr_schedule": None,
        # Share layers for value function. If you set this to True, it's important
        # to tune vf_loss_coeff.
        "vf_share_layers": False,
        # Coefficient of the value function loss. IMPORTANT: you must tune this if
        # you set vf_share_layers: True.
        "vf_loss_coeff": 1.0,
        # Coefficient of the entropy regularizer.
        "entropy_coeff": 0.0,
        # Decay schedule for the entropy regularizer.
        "entropy_coeff_schedule": None,
        # PPO clip parameter.
        "clip_param": 0.2,
        # Clip param for the value function. Note that this is sensitive to the
        # scale of the rewards. If your expected V is large, increase this.
        "vf_clip_param": 10000.0,
        # If specified, clip the global norm of gradients by this amount.
        "grad_clip": None,
        # Target value for KL divergence.
        "kl_target": 0.01,
        # Whether to rollout "complete_episodes" or "truncate_episodes".
        "batch_mode": "truncate_episodes",
        # Which observation filter to apply to the observation.
        "observation_filter": "NoFilter",
        # Uses the sync samples optimizer instead of the multi-gpu one. This is
        # usually slower, but you might want to try it if you run into issues with
        # the default optimizer.
        "simple_optimizer": False,
    }
