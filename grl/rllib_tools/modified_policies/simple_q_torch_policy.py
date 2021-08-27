"""PyTorch policy class used for Simple Q-Learning"""

import logging
from typing import Dict, Tuple

import gym
import ray
from ray.rllib.agents.dqn.simple_q_tf_policy import Q_SCOPE, Q_TARGET_SCOPE
from ray.rllib.agents.dqn.simple_q_tf_policy import (compute_q_values, get_distribution_inputs_and_class)
from ray.rllib.models import ModelCatalog
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType, TrainerConfigDict

from grl.rllib_tools.modified_policies.safe_set_weights_policy_mixin import SafeSetWeightsPolicyMixin

torch, nn = try_import_torch()
F = None
if nn:
    F = nn.functional
logger = logging.getLogger(__name__)


class TargetNetworkMixin:
    """Assign the `update_target` method to the SimpleQTorchPolicy

    The function is called every `target_network_update_freq` steps by the
    master learner.
    """

    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, config: TrainerConfigDict):
        def do_update():
            # Update_target_fn will be called periodically to copy Q network to
            # target Q network.
            assert len(self.q_func_vars) == len(self.target_q_func_vars), \
                (self.q_func_vars, self.target_q_func_vars)
            self.target_q_model.load_state_dict(self.q_model.state_dict())

        self.update_target = do_update


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


def build_q_losses(policy: Policy, model, dist_class,
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
    # q network evaluation
    q_t = compute_q_values(
        policy,
        policy.q_model,
        train_batch[SampleBatch.CUR_OBS],
        explore=False,
        is_training=True)

    # target q network evalution
    q_tp1 = compute_q_values(
        policy,
        policy.target_q_model,
        train_batch[SampleBatch.NEXT_OBS],
        explore=False,
        is_training=True)

    # q scores for actions which we know were selected in the given state.
    one_hot_selection = F.one_hot(train_batch[SampleBatch.ACTIONS].long(),
                                  policy.action_space.n)
    q_t_selected = torch.sum(q_t * one_hot_selection, 1)

    # compute estimate of best possible value starting from state at t + 1
    dones = train_batch[SampleBatch.DONES].float()
    q_tp1_best_one_hot_selection = F.one_hot(
        torch.argmax(q_tp1, 1), policy.action_space.n)
    q_tp1_best = torch.sum(q_tp1 * q_tp1_best_one_hot_selection, 1)
    q_tp1_best_masked = (1.0 - dones) * q_tp1_best

    # compute RHS of bellman equation
    q_t_selected_target = (train_batch[SampleBatch.REWARDS] +
                           policy.config["gamma"] * q_tp1_best_masked)

    # Compute the error (Square/Huber).
    td_error = q_t_selected - q_t_selected_target.detach()
    # loss = torch.mean(huber_loss(td_error)) # NFSP on Kuhn/Leduc poker fails with huber_loss

    loss = F.mse_loss(input=q_t_selected, target=q_t_selected_target.detach())

    # save TD error as an attribute for outside access
    policy.td_error = td_error
    policy.loss = loss
    return loss


def setup_late_mixins(policy: Policy, obs_space: gym.spaces.Space,
                      action_space: gym.spaces.Space,
                      config: TrainerConfigDict) -> None:
    """Call all mixin classes' constructors before SimpleQTorchPolicy
    initialization.

    Args:
        policy (Policy): The Policy object.
        obs_space (gym.spaces.Space): The Policy's observation space.
        action_space (gym.spaces.Space): The Policy's action space.
        config (TrainerConfigDict): The Policy's config.
    """
    TargetNetworkMixin.__init__(policy, obs_space, action_space, config)


SimpleQTorchPolicyPatched = build_torch_policy(
    name="SimpleQPolicy",
    loss_fn=build_q_losses,
    get_default_config=lambda: ray.rllib.agents.dqn.dqn.DEFAULT_CONFIG,
    after_init=setup_late_mixins,
    extra_action_out_fn=_simple_dqn_extra_action_out_fn,
    make_model_and_action_dist=_build_q_model_and_distribution,
    mixins=[TargetNetworkMixin, SafeSetWeightsPolicyMixin],
    action_distribution_fn=get_distribution_inputs_and_class,
    extra_learn_fetches_fn=lambda policy: {"td_error": policy.td_error, "loss": policy.loss},
)
