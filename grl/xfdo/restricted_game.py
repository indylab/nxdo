from typing import Tuple, Dict
from copy import deepcopy

from gym.spaces import Discrete

from grl.xfdo.action_space_conversion import RestrictedToBaseGameActionSpaceConverter

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict, AgentID

RESTRICTED_GAME = "restricted_game"


class RestrictedGame(MultiAgentEnv):

    def __init__(self, env_config: dict):
        self.base_env: MultiAgentEnv = env_config["create_env_fn"]()

        self._use_delegate_policy_exploration = env_config.get("use_delegate_policy_exploration", False)
        self._clip_base_game_actions = env_config.get("clip_base_game_actions", False)
        self._raise_if_no_restricted_players = env_config.get("raise_if_no_restricted_players", True)

        self.agents_to_action_converters = {}

        self.observation_space = self.base_env.observation_space
        self.base_observation_space = self.base_env.observation_space
        self.base_action_space = self.base_env.action_space

        self._agents_to_current_obs = {}

    def set_action_conversion(self, agent_id: AgentID, converter: RestrictedToBaseGameActionSpaceConverter):
        self.agents_to_action_converters[agent_id] = converter

    def reset(self) -> MultiAgentDict:
        if self._raise_if_no_restricted_players and len(self.agents_to_action_converters) == 0:
            raise ValueError("Restricted environment reset with no restricted players.")

        obs_dict = self.base_env.reset()
        self._agents_to_current_obs = deepcopy(obs_dict)
        return obs_dict

    def step(self, action_dict: MultiAgentDict) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:

        processed_action_dict = {}
        for agent_id, action in action_dict.items():
            if agent_id in self.agents_to_action_converters:
                convertor: RestrictedToBaseGameActionSpaceConverter = self.agents_to_action_converters[agent_id]
                base_game_action, _, _ = convertor.get_base_game_action(
                    obs=self._agents_to_current_obs[agent_id],
                    restricted_game_action=action,
                    use_delegate_policy_exploration=self._use_delegate_policy_exploration,
                    clip_base_game_actions=self._clip_base_game_actions,
                    delegate_policy_state=None)
                processed_action_dict[agent_id] = base_game_action
            else:
                processed_action_dict[agent_id] = action

        obs, rews, dones, infos = self.base_env.step(action_dict=processed_action_dict)

        for agent_id, observation in obs.items():
            self._agents_to_current_obs[agent_id] = observation

        return obs, rews, dones, infos
