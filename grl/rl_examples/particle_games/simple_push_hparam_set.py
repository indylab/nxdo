

import numpy as np
from gym.spaces import Box, Discrete, Space

from ray.rllib.utils import merge_dicts
from ray.rllib.models import MODEL_DEFAULTS
from ray.tune import grid_search, uniform, loguniform, sample_from, choice


def simple_push_sac_hparam_search(action_space: Space):
    if isinstance(action_space, Discrete):
        default_target_entropy = np.array(-np.log(1.0 / action_space.n), dtype=np.float32)
    else:
        default_target_entropy = -np.prod(action_space.shape)

    return {
        # Smooth metrics over this many episodes.
        "metrics_smoothing_episodes": 1000,
        "framework": "torch",
        # RL Algo Specific

        "initial_alpha": grid_search([1.0, 0.1, 0.01]),
        "target_entropy": grid_search([coeff/10.0 * default_target_entropy for coeff in range(0, 11)]),
        "Q_model":{
            "fcnet_activation": "relu",
            "fcnet_hiddens": grid_search(
                [[64],
                 [128],
                 [128, 128],
                 [128, 128, 128],
                 [256, 256, 256]]),
        },
        # Model options for the policy function.
        "policy_model": {
            "fcnet_activation": "relu",
            "fcnet_hiddens": sample_from(lambda spec: spec.config.Q_model.fcnet_hiddens),
        },
        "optimization": grid_search([{
            "actor_learning_rate": lr,
            "critic_learning_rate": lr,
            "entropy_learning_rate": lr,
        } for lr in [3e-2, 3e-3, 3e-4]]),

        "normalize_actions": False,
        "train_batch_size": grid_search([256, 1024, 2048, 4096]),
        "rollout_fragment_length": 100,
        "model": merge_dicts(MODEL_DEFAULTS, {}),
    }


def trial_name_creator(trial):
    config = trial.config
    print(f"trial keys: {list(config.keys())}")
    init_alpha = config["initial_alpha"]
    trgt_ent = config["target_entropy"]
    hiddens = "_".join(str(h) for h in config["policy_model"]["fcnet_hiddens"])
    lr = config["optimization"]["actor_learning_rate"]
    btch_sz = config["train_batch_size"]
    return f"init_alpha_{init_alpha}_trgt_ent_{trgt_ent}_layers_{hiddens}_lr_{lr}_btch_sz_{btch_sz}"
