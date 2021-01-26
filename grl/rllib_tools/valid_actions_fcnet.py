import logging
import numpy as np
from typing import Type

from gym.spaces import Box

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, \
    normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

from ray.rllib.models.torch.fcnet import FullyConnectedNetwork

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

ILLEGAL_ACTION_LOGITS_PENALTY = -1e9


def get_valid_action_fcn_class(obs_len: int, action_space_n: int, dummy_actions_multiplier: int = 1) -> Type[FullyConnectedNetwork]:

    class ValidActionFullyConnectedNetwork(FullyConnectedNetwork, nn.Module):

        @override(FullyConnectedNetwork)
        def __init__(self, obs_space, action_space, num_outputs, model_config,
                     name):
            obs_space = Box(low=0.0, high=1.0, shape=(obs_len,))
            FullyConnectedNetwork.__init__(self, obs_space=obs_space, action_space=action_space,
                                           num_outputs=num_outputs, model_config=model_config, name=name)

        @override(FullyConnectedNetwork)
        def forward(self, input_dict, state, seq_lens):
            obs = input_dict["obs_flat"].float()

            non_dummy_action_space_n = action_space_n // dummy_actions_multiplier

            assert obs.shape[1] == obs_len + non_dummy_action_space_n, \
                f"obs shape with valid action fc net is {obs.shape}\n" \
                f"obs_len without actions: {obs_len}\n" \
                f"non_dummy_action_space_n: {non_dummy_action_space_n}\n" \
                f"action space n: {action_space_n}\n" \
                f"obs: {obs}"

            # print(f"torch obs: {obs}")
            obs = obs[:, :-non_dummy_action_space_n]
            self.valid_actions_mask = input_dict['obs_flat'][:, -non_dummy_action_space_n:].repeat(1, dummy_actions_multiplier)
            # print(f"amt: {-VALID_ACTIONS_SHAPES[LEDUC_POKER][0]}")

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

    return ValidActionFullyConnectedNetwork
