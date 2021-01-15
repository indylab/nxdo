
import logging
from typing import Dict, Tuple, Type, List
import numpy as np
import gym
import ray
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchCategorical, \
    TorchDistributionWrapper
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import huber_loss
from ray.rllib.utils.typing import TensorType, TrainerConfigDict


from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import (Categorical,
                                                TFActionDistribution)
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.policy import Policy
from ray.rllib.policy.dynamic_tf_policy import DynamicTFPolicy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.tf_ops import huber_loss, make_tf_callable
from ray.rllib.utils.typing import TensorType, TrainerConfigDict

import grl

AVG_POL_SCOPE = "avg_pol"

torch, nn = try_import_torch()
F = None
if nn:
    F = nn.functional
logger = logging.getLogger(__name__)


def compute_policy_logits(policy: Policy,
                     model: ModelV2,
                     obs: TensorType,
                     is_training=None) -> TensorType:
    model_out, _ = model({
        SampleBatch.CUR_OBS: obs,
        "is_training": is_training
        if is_training is not None else policy._get_is_training_placeholder(),
    }, [], None)

    return model_out


def get_distribution_inputs_and_class(
        policy: Policy,
        model: ModelV2,
        obs_batch: TensorType,
        *,
        is_training=True,
        **kwargs) -> Tuple[TensorType, type, List[TensorType]]:
    """Build the action distribution"""
    logits = compute_policy_logits(policy, model, obs_batch, is_training)
    logits = logits[0] if isinstance(logits, tuple) else logits

    policy.logits = logits
    return policy.logits, (TorchCategorical
                           if policy.config["framework"] == "torch" else
                           Categorical), []  # state-outs


def build_avg_model_and_distribution(
        policy: Policy, obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        config: TrainerConfigDict) -> Tuple[ModelV2, Type[TorchDistributionWrapper]]:

    if not isinstance(action_space, gym.spaces.Discrete):
        raise UnsupportedSpaceException(f"Action space {action_space} is not supported for NFSP.")

    policy.avg_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=action_space.n,
        model_config=config["model"],
        framework=config["framework"],
        name=AVG_POL_SCOPE)

    policy.avg_func_vars = policy.avg_model.variables()

    return policy.avg_model, TorchCategorical



def build_supervised_learning_loss(policy: Policy, model: ModelV2, dist_class: Type[TorchDistributionWrapper],
                   train_batch: SampleBatch) -> TensorType:
    """Constructs the loss for SimpleQTorchPolicy.

    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[ActionDistribution]): The action distribution class.
        train_batch (SampleBatch): The training data.

    Returns:
        TensorType: A single loss tensor.
    """
    logits_t = compute_policy_logits(policy=policy,
                                     model=policy.avg_model,
                                     obs=train_batch[SampleBatch.CUR_OBS],
                                     is_training=True)

    action_targets_t = train_batch[SampleBatch.ACTIONS].long()

    policy.loss = F.cross_entropy(input=logits_t, target=action_targets_t)

    return policy.loss

def behaviour_logits_fetches(
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
        "action_probs": policy.action_probs,
        "behaviour_logits": policy.logits,
    }

# actions, logp, state_out = self.action_sampler_fn(
#                 self,
#                 self.model,
#                 input_dict,
#                 state_out,
#                 explore=explore,
#                 timestep=timestep)


def action_sampler(policy, model, input_dict, state, explore, timestep):
    obs: np.ndarray = input_dict['obs']
    is_training = False
    logits = compute_policy_logits(policy, model, obs, is_training)
    logits = logits[0] if isinstance(logits, tuple) else logits
    action_probs_batch = F.softmax(logits, dim=1)
    policy.logits = logits
    policy.action_probs = action_probs_batch
    # print(f"probs: {action_probs_batch}")

    actions = []
    logps = []
    for action_probs in action_probs_batch.cpu().detach().numpy():
        action = np.random.choice(range(0, len(action_probs)), p=action_probs)
        logp = np.log(action_probs[action])
        # print(f"action: {action}, logp: {logp}")
        actions.append(action)
        logps.append(logp)
    state_out = state
    return np.asarray(actions, dtype=np.int32), np.asarray(logps, dtype=np.float32), state_out

def sgd_optimizer(policy: Policy,
                   config: TrainerConfigDict) -> "torch.optim.Optimizer":
    return torch.optim.SGD(
        policy.avg_func_vars, lr=policy.config["lr"])


def build_avg_policy_stats(policy: Policy, batch) -> Dict[str, TensorType]:
    return {"loss": policy.loss}

NFSPTorchAveragePolicy = build_torch_policy(
    name="NFSPAveragePolicy",
    extra_action_out_fn=behaviour_logits_fetches,
    loss_fn=build_supervised_learning_loss,
    get_default_config=lambda: grl.nfsp_rllib.nfsp.DEFAULT_CONFIG,
    make_model_and_action_dist=build_avg_model_and_distribution,
    action_sampler_fn=action_sampler,
    extra_learn_fetches_fn=lambda policy: {"sl_loss": policy.loss},
    optimizer_fn=sgd_optimizer,
    stats_fn=build_avg_policy_stats,
    # action_distribution_fn=get_distribution_inputs_and_class,
)
