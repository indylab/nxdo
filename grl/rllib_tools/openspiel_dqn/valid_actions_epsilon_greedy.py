import numpy as np
from typing import Union, Optional

from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils.annotations import override
from ray.rllib.utils.exploration.exploration import Exploration, TensorType
from ray.rllib.utils.framework import try_import_tf, try_import_torch, \
    get_variable
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.schedules import Schedule, PiecewiseSchedule
from ray.rllib.utils.torch_ops import FLOAT_MIN

from grl.rllib_tools.openspiel_dqn.valid_actions_fcnet import ILLEGAL_ACTION_LOGITS_PENALTY

tf1, tf, tfv = try_import_tf()
torch, _ = try_import_torch()

import torch

class ValidActionsEpsilonGreedy(Exploration):
    """Epsilon-greedy Exploration class that produces exploration actions.

    When given a Model's output and a current epsilon value (based on some
    Schedule), it produces a random action (if rand(1) < eps) or
    uses the model-computed one (if rand(1) >= eps).
    """

    def __init__(self,
                 action_space,
                 *,
                 framework: str,
                 initial_epsilon: float = 1.0,
                 final_epsilon: float = 0.05,
                 epsilon_timesteps: int = int(1e5),
                 epsilon_schedule: Optional[Schedule] = None,
                 **kwargs):
        """Create an EpsilonGreedy exploration class.

        Args:
            initial_epsilon (float): The initial epsilon value to use.
            final_epsilon (float): The final epsilon value to use.
            epsilon_timesteps (int): The time step after which epsilon should
                always be `final_epsilon`.
            epsilon_schedule (Optional[Schedule]): An optional Schedule object
                to use (instead of constructing one from the given parameters).
        """
        assert framework is not None
        super().__init__(
            action_space=action_space, framework=framework, **kwargs)

        self.epsilon_schedule = \
            from_config(Schedule, epsilon_schedule, framework=framework) or \
            PiecewiseSchedule(
                endpoints=[
                    (0, initial_epsilon), (epsilon_timesteps, final_epsilon)],
                outside_value=final_epsilon,
                framework=self.framework)

        # The current timestep value (tf-var or python int).
        self.last_timestep = get_variable(
            np.array(0, np.int64),
            framework=framework,
            tf_name="timestep",
            dtype=np.int64)

        # Build the tf-info-op.
        if self.framework in ["tf2", "tf", "tfe"]:
            self._tf_info_op = self.get_info()

    @override(Exploration)
    def get_exploration_action(self,
                               *,
                               action_distribution: ActionDistribution,
                               timestep: Union[int, TensorType],
                               explore: bool = True):

        q_values = action_distribution.inputs

        return self._get_torch_exploration_action(q_values, explore,
                                                      timestep)

    def _get_torch_exploration_action(self, q_values: TensorType,
                                      explore: bool,
                                      timestep: Union[int, TensorType]):
        """Torch method to produce an epsilon exploration action.

        Args:
            q_values (Tensor): The Q-values coming from some Q-model.

        Returns:
            torch.Tensor: The exploration-action.
        """
        self.last_timestep = timestep
        _, exploit_action = torch.max(q_values, 1)
        action_logp = torch.zeros_like(exploit_action)

        # Explore.
        if explore:
            # Get the current epsilon.
            epsilon = self.epsilon_schedule(self.last_timestep)
            batch_size = q_values.size()[0]
            # Mask out actions, whose Q-values are less than ILLEGAL_ACTION_LOGITS_PENALTY, so that we don't
            # even consider them for exploration.
            random_valid_action_logits = torch.where(
                q_values <= ILLEGAL_ACTION_LOGITS_PENALTY,
                torch.ones_like(q_values) * 0.0, torch.ones_like(q_values))
            # A random action.
            random_actions = torch.squeeze(
                torch.multinomial(random_valid_action_logits, 1), axis=1)
            # Pick either random or greedy.
            action = torch.where(
                torch.empty(
                    (batch_size, )).uniform_().to(self.device) < epsilon,
                random_actions, exploit_action)

            return action, action_logp
        # Return the deterministic "sample" (argmax) over the logits.
        else:
            return exploit_action, action_logp

    @override(Exploration)
    def get_info(self, sess: Optional["tf.Session"] = None):
        if sess:
            return sess.run(self._tf_info_op)
        eps = self.epsilon_schedule(self.last_timestep)
        return {"cur_epsilon": eps}
