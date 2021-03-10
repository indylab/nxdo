from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.utils import merge_dicts

# To use different params, please make define a mutation of these master parameters
# instead of changing the constants already defined here.


GRL_DEFAULT_OPENSPIEL_POKER_DQN_PARAMS = {
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
        "epsilon_timesteps": int(3e7),  # Timesteps over which to anneal epsilon.
    },
    # Switch to greedy actions in evaluation workers.
    "evaluation_config": {
        "explore": False,
    },
    "explore": True,

    # Update the target network every `target_network_update_freq` steps.
    "target_network_update_freq": 10000,

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

GRL_DEFAULT_POKER_PPO_PARAMS = {
    "framework": "torch",
    # Should use a critic as a baseline (otherwise don't use value baseline;
    # required for using GAE).
    "use_critic": True,
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "use_gae": True,
    # The GAE(lambda) parameter.
    "lambda": 1.0,
    # Initial coefficient for KL divergence.
    "kl_coeff": 0.2,
    # Size of batches collected from each worker.
    "rollout_fragment_length": 256,
    # Number of timesteps collected for each SGD round. This defines the size
    # of each SGD epoch.
    "train_batch_size": 2048,
    # Total SGD batch size across all devices for SGD. This defines the
    # minibatch size within each epoch.
    "sgd_minibatch_size": 128,
    # Whether to shuffle sequences in the batch when training (recommended).
    "shuffle_sequences": True,
    # Number of SGD iterations in each outer loop (i.e., number of epochs to
    # execute per train batch).
    "num_sgd_iter": 30,
    # Stepsize of SGD.
    "lr": 5e-5,
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
    "clip_param": 0.3,
    # Clip param for the value function. Note that this is sensitive to the
    # scale of the rewards. If your expected V is large, increase this.
    "vf_clip_param": 10.0,
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
    # Whether to fake GPUs (using CPUs).
    # Set this to True for debugging on non-GPU machines (set `num_gpus` > 0).
    "_fake_gpus": False,
    # Switch on Trajectory View API for PPO by default.
    # NOTE: Only supported for PyTorch so far.
    "_use_trajectory_view_api": True,

    # Prevent iterations from going lower than this time span
    "min_iter_time_s": 0,
    # Minimum env steps to optimize for per train call. This value does
    # not affect learning (JB: this is a lie!), only the length of train iterations.
    "timesteps_per_iteration": 0,

    "num_envs_per_worker": 1,

    "model": merge_dicts(MODEL_DEFAULTS, {
        "fcnet_activation": "relu",
        "fcnet_hiddens": [128],
        "custom_model": None,
    }),
}
