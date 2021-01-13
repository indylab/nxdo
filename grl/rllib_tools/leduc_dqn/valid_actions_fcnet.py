import logging
import numpy as np

from gym.spaces import Box

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, \
    normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

from ray.rllib.models.torch.fcnet import FullyConnectedNetwork

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

from grl.rl_apps.kuhn_poker_p2sro.poker_multi_agent_env import VALID_ACTIONS_SHAPES, LEDUC_POKER, OBS_SHAPES

ILLEGAL_ACTION_LOGITS_PENALTY = -1e9


class LeducDQNFullyConnectedNetwork(FullyConnectedNetwork, nn.Module):

    @override(FullyConnectedNetwork)
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        obs_space = Box(low=0.0, high=1.0, shape=OBS_SHAPES[LEDUC_POKER])

        FullyConnectedNetwork.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)

    @override(FullyConnectedNetwork)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()

        assert obs.shape[1] == OBS_SHAPES[LEDUC_POKER][0] + VALID_ACTIONS_SHAPES[LEDUC_POKER][0], \
            f"obs shape with leduc dqn fc net is {obs.shape}"

        # print(f"torch obs: {obs}")
        obs = obs[:, :-VALID_ACTIONS_SHAPES[LEDUC_POKER][0]]
        self.valid_actions_mask = input_dict['obs_flat'][:, -VALID_ACTIONS_SHAPES[LEDUC_POKER][0]:]
        # print(f"amt: {-VALID_ACTIONS_SHAPES[LEDUC_POKER][0]}")
        # print(f"torch mask: {self.valid_actions_mask}")

        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)
        logits = self._logits(self._features) if self._logits else \
            self._features
        if self.free_log_std:
            raise NotImplementedError
            # logits = self._append_free_log_std(logits)

        illegal_actions = 1 - self.valid_actions_mask
        illegal_logit_penalties = illegal_actions * ILLEGAL_ACTION_LOGITS_PENALTY

        masked_logits = logits + illegal_logit_penalties

        # print(logits)
        # print(self.valid_actions_mask)
        # print(masked_logits)

        return masked_logits, state

    @override(FullyConnectedNetwork)
    def value_function(self):
        raise NotImplementedError

        # assert self._features is not None, "must call forward() first"
        # if self._value_branch_separate:
        #     return self._value_branch(
        #         self._value_branch_separate(self._last_flat_in)).squeeze(1)
        # else:
        #     return self._value_branch(self._features).squeeze(1)
