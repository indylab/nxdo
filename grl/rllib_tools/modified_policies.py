from typing import List, Dict, Tuple
import gym

import ray
from ray.rllib.utils import try_import_torch

torch, _ = try_import_torch()


from ray.rllib.agents.sac import SACTorchPolicy
from ray.rllib.utils.typing import TensorType
from ray.rllib.policy import Policy
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.agents.dqn.simple_q_torch_policy import SimpleQTorchPolicy

from ray.rllib.utils.typing import TensorType, TrainerConfigDict
from ray.rllib.agents.dqn.simple_q_tf_policy import Q_SCOPE, Q_TARGET_SCOPE
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_action_dist import TorchCategorical

def _simple_dqn_extra_action_out_fn(policy: Policy, input_dict, state_batches, model,
                        action_dist: ActionDistribution) -> Dict[str, TensorType]:
    action = action_dist.deterministic_sample()
    action_probs = torch.zeros_like(policy.q_values)
    action_probs[0][action[0]] = 1.0
    return {"q_values": policy.q_values, "action_probs": action_probs}

def _build_q_models(policy: Policy, obs_space: gym.spaces.Space,
                   action_space: gym.spaces.Space,
                   config: TrainerConfigDict) -> ModelV2:
    """Build q_model and target_q_model for Simple Q learning

    Note that this function works for both Tensorflow and PyTorch.

    Args:
        policy (Policy): The Policy, which will use the model for optimization.
        obs_space (gym.spaces.Space): The policy's observation space.
        action_space (gym.spaces.Space): The policy's action space.
        config (TrainerConfigDict):

    Returns:
        ModelV2: The Model for the Policy to use.
            Note: The target q model will not be returned, just assigned to
            `policy.target_q_model`.
    """
    if not isinstance(action_space, gym.spaces.Discrete):
        raise UnsupportedSpaceException(
            "Action space {} is not supported for DQN.".format(action_space))

    policy.q_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=action_space.n,
        model_config=config["model"],
        framework=config["framework"],
        name=Q_SCOPE)
    if torch.cuda.is_available():
        policy.q_model = policy.q_model.to("cuda")

    policy.target_q_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=action_space.n,
        model_config=config["model"],
        framework=config["framework"],
        name=Q_TARGET_SCOPE)
    if torch.cuda.is_available():
        policy.target_q_model = policy.target_q_model.to("cuda")

    policy.q_func_vars = policy.q_model.variables()
    policy.target_q_func_vars = policy.target_q_model.variables()

    return policy.q_model

def _build_q_model_and_distribution(
        policy: Policy, obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        config: TrainerConfigDict) -> Tuple[ModelV2, TorchDistributionWrapper]:
    return _build_q_models(policy, obs_space, action_space, config), \
        TorchCategorical

SimpleQTorchPolicyPatched = SimpleQTorchPolicy.with_updates(
    extra_action_out_fn=_simple_dqn_extra_action_out_fn,
    make_model_and_action_dist=_build_q_model_and_distribution,
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