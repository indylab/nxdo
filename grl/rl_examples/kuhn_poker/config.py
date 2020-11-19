import numpy as np
from gym.spaces import Discrete

from ray.rllib.utils import merge_dicts
from ray.rllib.models import MODEL_DEFAULTS

def kuhn_sac_params(action_space: Discrete):
    return {
        # Smooth metrics over this many episodes.
        "metrics_smoothing_episodes": 1000,

        "framework": "torch",
        # RL Algo Specific
        "initial_alpha": 0.0,
        "target_entropy": 0,
        "train_batch_size": 200,
        "rollout_fragment_length": 10,
        "normalize_actions": False,
        "model": merge_dicts(MODEL_DEFAULTS, {}),

        "Q_model": {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [60, 60],
        },
        # Model options for the policy function.
        "policy_model": {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [60, 60],
        },

        "use_state_preprocessor": False,
        "optimization": {
            "actor_learning_rate": 1e-2,
            "critic_learning_rate": 1e-2,
            "entropy_learning_rate": 1e-2,
        },

    }