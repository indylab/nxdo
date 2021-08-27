from typing import Dict, Callable, List, Tuple

import numpy as np
from gym.spaces import Box
from open_spiel.python.policy import TabularPolicy
from pyspiel import Game as OpenSpielGame
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import MultiAgentDict, AgentID

from grl.rl_apps.psro.poker_utils import softmax
from grl.utils.strategy_spec import StrategySpec
from grl.envs.valid_actions_multi_agent_env import ValidActionsMultiAgentEnv

class AgentRestrictedGameOpenSpielObsConversions:

    def __init__(self,
                 orig_obs_to_restricted_game_obs: dict,
                 orig_obs_to_info_state_vector: dict,
                 orig_obs_to_restricted_game_valid_actions_mask: dict):
        self.orig_obs_to_restricted_game_obs = orig_obs_to_restricted_game_obs
        self.orig_obs_to_info_state_vector = orig_obs_to_info_state_vector
        self.orig_obs_to_restricted_game_valid_actions_mask = orig_obs_to_restricted_game_valid_actions_mask


def get_restricted_game_obs_conversions(
        player: int,
        delegate_policy: Policy,
        policy_specs: List[StrategySpec],
        load_policy_spec_fn: Callable[[Policy, StrategySpec], None],
        tmp_base_env: MultiAgentEnv) -> AgentRestrictedGameOpenSpielObsConversions:
    openspiel_game: OpenSpielGame = tmp_base_env.openspiel_env.game

    empty_tabular_policy = TabularPolicy(openspiel_game)

    orig_obs_to_restricted_game_valid_actions_mask = {}
    orig_obs_to_info_state_vector = {}

    for policy_spec in policy_specs:
        load_policy_spec_fn(delegate_policy, policy_spec)

        for state_index, state in enumerate(empty_tabular_policy.states):
            assert state.current_player() in [0, 1], state.current_player()

            valid_actions_mask = state.legal_actions_mask()
            info_state_vector = state.information_state_tensor()

            if openspiel_game.get_type().short_name in ["leduc_poker", "oshi_zumo", "oshi_zumo_tiny",
                                                        "universal_poker"]:
                # Observation includes both the info_state and legal actions, but agent isn't forced to take legal actions.
                # Taking an illegal action will result in a random legal action being played.
                # Allows easy compatibility with standard RL implementations for small action-space games like this one.
                obs = np.concatenate(
                    (np.asarray(info_state_vector, dtype=np.float32), np.asarray(valid_actions_mask, dtype=np.float32)),
                    axis=0)
            else:
                obs = np.asarray(info_state_vector, dtype=np.float32)

            _, _, action_info = delegate_policy.compute_single_action(obs=obs, state=[], explore=False)

            action_probs = None
            for key in ['policy_targets', 'action_probs']:
                if key in action_info:
                    action_probs = action_info[key]
                    break
            if action_probs is None:
                action_logits = action_info['behaviour_logits']
                action_probs = softmax(action_logits)
            original_action_probs = action_probs.copy()

            # verify all probability is on a single action
            assert sum(original_action_probs == 1.0) == 1
            assert sum(original_action_probs == 0.0) == len(original_action_probs) - 1

            # verify action probs are valid
            if len(action_probs) > len(valid_actions_mask) and len(action_probs) % len(valid_actions_mask) == 0:
                # we may be using a dummy action variant of poker
                dummy_action_probs = action_probs.copy()
                action_probs = np.zeros_like(valid_actions_mask, dtype=np.float64)
                for i, action_prob in enumerate(dummy_action_probs):
                    action_probs[i % len(valid_actions_mask)] += action_prob
                assert np.isclose(sum(action_probs), 1.0), sum(action_probs)
            assert np.isclose(sum(action_probs), 1.0)
            legal_action_probs = []
            valid_action_prob_sum = 0.0
            for idx in range(len(valid_actions_mask)):
                if valid_actions_mask[idx] == 1.0:
                    legal_action_probs.append(action_probs[idx])
                    valid_action_prob_sum += action_probs[idx]
            assert np.isclose(valid_action_prob_sum, 1.0), (
                action_probs, valid_actions_mask, action_info.get('behaviour_logits'))

            obs_key = tuple(obs)
            if obs_key not in orig_obs_to_info_state_vector:
                orig_obs_to_info_state_vector[obs_key] = np.asarray(info_state_vector, dtype=np.float32)

            if obs_key in orig_obs_to_restricted_game_valid_actions_mask:
                orig_obs_to_restricted_game_valid_actions_mask[obs_key] = np.clip(
                    orig_obs_to_restricted_game_valid_actions_mask[obs_key] + original_action_probs, a_max=1.0,
                    a_min=0.0)
            else:
                orig_obs_to_restricted_game_valid_actions_mask[obs_key] = original_action_probs.copy()

    assert set(orig_obs_to_info_state_vector.keys()) == set(orig_obs_to_restricted_game_valid_actions_mask.keys())
    orig_obs_to_restricted_game_obs = {}
    for orig_obs_keys in orig_obs_to_info_state_vector.keys():
        orig_info_state_vector = orig_obs_to_info_state_vector[orig_obs_keys]
        restricted_game_valid_actions_mask = orig_obs_to_restricted_game_valid_actions_mask[orig_obs_keys]
        orig_obs_to_restricted_game_obs[orig_obs_keys] = np.concatenate((
            np.asarray(orig_info_state_vector, dtype=np.float32),
            np.asarray(restricted_game_valid_actions_mask, dtype=np.float32)
        ), axis=0)

    return AgentRestrictedGameOpenSpielObsConversions(
        orig_obs_to_restricted_game_obs=orig_obs_to_restricted_game_obs,
        orig_obs_to_info_state_vector=orig_obs_to_info_state_vector,
        orig_obs_to_restricted_game_valid_actions_mask=orig_obs_to_restricted_game_valid_actions_mask
    )


class OpenSpielRestrictedGame(ValidActionsMultiAgentEnv):

    def __init__(self, env_config: dict):
        self.base_env: MultiAgentEnv = env_config["create_env_fn"]()

        self._clip_base_game_actions = env_config.get("clip_base_game_actions", False)
        self._raise_if_no_restricted_players = env_config.get("raise_if_no_restricted_players", True)

        self.agent_conversions: Dict[AgentID, AgentRestrictedGameOpenSpielObsConversions] = {}

        base_env_action_space_without_dummy = self.base_env.action_space.n / self.base_env.dummy_action_multiplier
        rstr_obs_len = (np.shape(list(self.base_env.reset().values())[0])[
                            0] - base_env_action_space_without_dummy) + self.base_env.action_space.n

        self.observation_space = Box(low=self.base_env.observation_space.low[0],
                                     high=self.base_env.observation_space.high[0],
                                     shape=(int(rstr_obs_len),)
                                     )
        self.base_observation_space = self.base_env.observation_space
        self.base_action_space = self.base_env.action_space
        self.action_space = self.base_env.action_space
        self.orig_observation_length = self.base_env.orig_observation_length

        self._agents_to_current_valid_actions_mask = {}

    def set_obs_conversion_dict(self, agent_id: AgentID, obs_conversions: AgentRestrictedGameOpenSpielObsConversions):
        self.agent_conversions[agent_id] = obs_conversions

    def _convert_obs_to_restricted_game(self, base_game_obs_dict: MultiAgentDict, dones):
        obs_dict_out = {}

        self._agents_to_current_valid_actions_mask = {agent: None for agent in range(2)}

        for agent_id, base_game_obs in base_game_obs_dict.items():
            if agent_id in self.agent_conversions:
                if not dones["__all__"]:
                    base_game_obs_as_tuple = tuple(base_game_obs)
                    try:
                        restricted_game_obs = self.agent_conversions[agent_id].orig_obs_to_restricted_game_obs[
                            base_game_obs_as_tuple]
                        # assert len(restricted_game_obs) == 90, "only needs to be true for 20x dummy leduc"
                    except KeyError:
                        assert isinstance(base_game_obs_as_tuple, tuple)
                        assert base_game_obs_as_tuple[0] == \
                               list(self.agent_conversions[agent_id].orig_obs_to_restricted_game_obs.keys())[0][
                                   0], f"key provided is {base_game_obs_as_tuple}\n agent id is {agent_id} \n example key is {list(self.agent_conversions[agent_id].orig_obs_to_restricted_game_obs.keys())[0]}"
                        assert len(base_game_obs_as_tuple) == len(
                            list(self.agent_conversions[agent_id].orig_obs_to_restricted_game_obs.keys())[
                                0]), f"{len(base_game_obs_as_tuple)} {len(list(self.agent_conversions[agent_id].orig_obs_to_restricted_game_obs.keys())[0])}"
                        print(
                            f"keys are: {self.agent_conversions[agent_id].orig_obs_to_restricted_game_obs.keys()}\n\nlooking for {base_game_obs_as_tuple}")
                        raise
                    self._agents_to_current_valid_actions_mask[agent_id] = \
                    self.agent_conversions[agent_id].orig_obs_to_restricted_game_valid_actions_mask[
                        base_game_obs_as_tuple]
                    obs_dict_out[agent_id] = restricted_game_obs
                else:
                    restricted_game_obs = np.zeros(shape=self.observation_space.shape, dtype=np.float32)
                    restricted_game_obs[:len(base_game_obs)] = base_game_obs
                    obs_dict_out[agent_id] = restricted_game_obs
            else:
                obs_dict_out[agent_id] = base_game_obs
        return obs_dict_out

    def reset(self) -> MultiAgentDict:
        if self._raise_if_no_restricted_players and len(self.agent_conversions) == 0:
            raise ValueError("Restricted environment reset with no restricted players.")
        obs_dict = self.base_env.reset()
        return self._convert_obs_to_restricted_game(base_game_obs_dict=obs_dict,
                                                    dones={0: False, 1: False, "__all__": False})

    def step(self, action_dict: MultiAgentDict) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:

        for agent_id, action in action_dict.items():
            if self._agents_to_current_valid_actions_mask[agent_id] is not None:
                assert self._agents_to_current_valid_actions_mask[agent_id][action] == 1.0, f"\nagent is {agent_id} " \
                                                                                            f"action is {action}" \
                                                                                            f"rstr valid_actions are {self._agents_to_current_valid_actions_mask[agent_id]}"

        base_obs_dict, rews, dones, infos = self.base_env.step(action_dict=action_dict)

        restricted_game_obs = self._convert_obs_to_restricted_game(base_game_obs_dict=base_obs_dict, dones=dones)

        return restricted_game_obs, rews, dones, infos
