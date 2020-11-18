from ray.rllib.utils import merge_dicts
from ray.rllib.models import MODEL_DEFAULTS

kuhn_sac_params = {
    "framework": "torch",
    # RL Algo Specific
    "target_entropy": 0.0,
    "train_batch_size": 200,
    "rollout_fragment_length": 10,
    "normalize_actions": False,
    "model": merge_dicts(MODEL_DEFAULTS, {})
}