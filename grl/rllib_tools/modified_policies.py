from typing import List, Dict

from ray.rllib.utils import try_import_torch

torch, _ = try_import_torch()

from ray.rllib.agents.sac import SACTorchPolicy
from ray.rllib.utils.typing import TensorType
from ray.rllib.policy import Policy
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.agents.dqn.simple_q_torch_policy import SimpleQTorchPolicy


def _simple_dqn_extra_action_out_fn(policy: Policy, input_dict, state_batches, model,
                        action_dist: ActionDistribution) -> Dict[str, TensorType]:
    action = action_dist.deterministic_sample()
    action_probs = torch.zeros_like(policy.q_values).long()
    action_probs[0][action[0]] = 1.0
    return {"q_values": policy.q_values, "action_probs": action_probs}


SimpleQTorchPolicyWithActionProbsOut = SimpleQTorchPolicy.with_updates(
    extra_action_out_fn=_simple_dqn_extra_action_out_fn
)


def _sac_behaviour_logits_fetches(
        policy: Policy, input_dict: Dict[str, TensorType],
        state_batches: List[TensorType], model: ModelV2,
        action_dist: TorchDistributionWrapper) -> Dict[str, TensorType]:
    """Defines extra fetches per action computation.

    Args:
        policy (Policy): The Policy to perform the extra action fetch on.
        input_dict (Dict[str, TensorType]): The input dict used for the action
            computing forward pass.
        state_batches (List[TensorType]): List of state tensors (empty for
            non-RNNs).
        model (ModelV2): The Model object of the Policy.
        action_dist (TorchDistributionWrapper): The instantiated distribution
            object, resulting from the model's outputs and the given
            distribution class.

    Returns:
        Dict[str, TensorType]: Dict with extra tf fetches to perform per
            action computation.
    """
    return {
        "behaviour_logits": policy.logits,
    }


SACTorchPolicyWithBehaviorLogitsOut = SACTorchPolicy.with_updates(
    extra_action_out_fn=_sac_behaviour_logits_fetches
)