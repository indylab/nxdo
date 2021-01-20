import random
import copy

import numpy as np
from gym.spaces import Discrete, Box, Dict
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from open_spiel.python.rl_environment import TimeStep, Environment

def with_base_config(base_config, extra_config):
    config = copy.deepcopy(base_config)
    config.update(extra_config)
    return config


OSHI_ZUMO = "oshi_zumo"

OSHI_ZUMO_OBS_LENGTH = 30

DEFAULT_CONFIG = {
    'version': OSHI_ZUMO,
    'fixed_players': True,
    'dummy_action_multiplier': 1,
    'continuous_action_space': False,
    'append_valid_actions_mask_to_obs': False,
}

class OshiZumoMultiAgentEnv(MultiAgentEnv):

    def __init__(self, env_config=None):
        env_config = with_base_config(base_config=DEFAULT_CONFIG, extra_config=env_config if env_config else {})
        self._fixed_players = env_config['fixed_players']
        self.game_version = env_config['version']

        if not isinstance(env_config['dummy_action_multiplier'], int) and env_config['dummy_action_multiplier'] > 0:
            raise ValueError("dummy_action_multiplier must be a positive non-zero int")
        self._dummy_action_multiplier = env_config['dummy_action_multiplier']
        self._continuous_action_space = env_config['continuous_action_space']
        self._append_valid_actions_mask_to_obs = env_config["append_valid_actions_mask_to_obs"]

        self.openspiel_env = Environment(game_name=self.game_version, discount=1.0)

        self.base_num_discrete_actions = self.openspiel_env.action_spec()["num_actions"]
        self.num_discrete_actions = int(self.base_num_discrete_actions * self._dummy_action_multiplier)
        self._base_action_space = Discrete(self.base_num_discrete_actions)

        if self._continuous_action_space:
            self.action_space = Box(low=-1, high=1, shape=1)
        else:
            self.action_space = Discrete(self.num_discrete_actions)

        if self._append_valid_actions_mask_to_obs:
            self.observation_length = self.openspiel_env.observation_spec()["info_state"][0] + self.num_discrete_actions
        else:
            self.observation_length = self.openspiel_env.observation_spec()["info_state"][0]

        self.observation_space = Box(low=0.0, high=1.0, shape=(self.observation_length,))

        self.curr_time_step: TimeStep = None
        self.player_map = None

    def _get_current_obs(self):

        done = self.curr_time_step.last()
        obs = {}
        if done:
            player_ids = [0, 1]
        else:
            curr_player_id = self.curr_time_step.observations["current_player"]
            player_ids = [curr_player_id]

        for player_id in player_ids:
            legal_actions = self.curr_time_step.observations["legal_actions"][player_id]
            legal_actions_mask = np.zeros(self.openspiel_env.action_spec()["num_actions"])
            legal_actions_mask[legal_actions] = 1.0

            info_state = self.curr_time_step.observations["info_state"][player_id]

            if self._append_valid_actions_mask_to_obs:
                # Observation includes both the info_state and legal actions, but agent isn't forced to take legal actions.
                # Taking an illegal action will result in a random legal action being played.
                # Allows easy compatibility with standard RL implementations for small action-space games like this one.
                obs[self.player_map(player_id)] = np.concatenate(
                    (np.asarray(info_state, dtype=np.float32), np.asarray(legal_actions_mask, dtype=np.float32)),
                    axis=0)
            else:
                obs[self.player_map(player_id)] = np.asarray(info_state, dtype=np.float32)

        return obs

    def reset(self):
        """Resets the env and returns observations from ready agents.

        Returns:
            obs (dict): New observations for each ready agent.
        """
        self.curr_time_step = self.openspiel_env.reset()

        if self._fixed_players:
            self.player_map = lambda p: p
        else:
            # swap player mapping in half of the games
            assert False, "debugging assert, ok to remove"
            self.player_map = random.choice((lambda p: p,
                                             lambda p: (1 - p)))

        return self._get_current_obs()

    def step(self, action_dict):
        """Returns observations from ready agents.

        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.

        Returns
        -------
            obs (dict): New observations for each ready agent.
            rewards (dict): Reward values for each ready agent. If the
                episode is just started, the value will be None.
            dones (dict): Done values for each ready agent. The special key
                "__all__" (required) is used to indicate env termination.
            infos (dict): Optional info values for each agent id.
        """
        curr_player_id = self.curr_time_step.observations["current_player"]
        legal_actions = self.curr_time_step.observations["legal_actions"][curr_player_id]

        player_action = action_dict[self.player_map(curr_player_id)]
        orig_player_action = player_action

        if self._continuous_action_space:
            # player action is between -1 and 1, normalize to 0 and 1 and then quantize to a discrete action
            player_action = (player_action / 2.0) + 1.0
            assert 0.0 - 1e-9 <= player_action <= 1.0 + 1e-9
            # place discrete actions in [0, 1] and find closest corresponding discrete action to player action
            nearest_discrete_action = min(range(0, self.num_discrete_actions),
                                          key=lambda x: abs(x/(self.num_discrete_actions - 1) - player_action))
            # player action is now a discrete action
            player_action = nearest_discrete_action

        if self._dummy_action_multiplier != 1:
            # extended dummy action space is just the base discrete actions repeated multiple times
            # convert to the base discrete action space.
            player_action = player_action % self.base_num_discrete_actions

        if player_action not in self._base_action_space:
            raise ValueError("Processed player action isn't in the base action space.\n"
                             f"orig action: {orig_player_action}\n"
                             f"processed action: {player_action}\n"
                             f"action space: {self.action_space}\n"
                             f"base action space: {self._base_action_space}")

        # If action is illegal, do a random legal action instead
        if player_action not in legal_actions:
            raise ValueError("illegal actions are now not allowed")
            # player_action = random.choice(legal_actions)
            # if self._apply_penalty_for_invalid_actions:
            #     self._invalid_action_penalties[curr_player_id] = True

        self.curr_time_step = self.openspiel_env.step([player_action])

        new_curr_player_id = self.curr_time_step.observations["current_player"]
        obs = self._get_current_obs()
        done = self.curr_time_step.last()
        dones = {self.player_map(new_curr_player_id): done, "__all__": done}

        if done:
            # dones = {0: True, 1: True, "__all__": True}

            rewards = {self.player_map(0): self.curr_time_step.rewards[0],
                       self.player_map(1): self.curr_time_step.rewards[1]}

            assert self.curr_time_step.rewards[0] == -self.curr_time_step.rewards[1]

            infos = {0: {}, 1: {}}

            infos[self.player_map(0)]['game_result_was_invalid'] = False
            infos[self.player_map(1)]['game_result_was_invalid'] = False

            assert sum(self.curr_time_step.rewards) == 0.0, "curr_time_step rewards in are terminal state are {} (they should sum to zero)".format(self.curr_time_step.rewards)

            infos[self.player_map(0)]['rewards'] = self.curr_time_step.rewards[0]
            infos[self.player_map(1)]['rewards'] = self.curr_time_step.rewards[1]

            if self.curr_time_step.rewards[0] > 0:
                infos[self.player_map(0)]['game_result'] = 'won'
                infos[self.player_map(1)]['game_result'] = 'lost'
            elif self.curr_time_step.rewards[1] > 0:
                infos[self.player_map(1)]['game_result'] = 'won'
                infos[self.player_map(0)]['game_result'] = 'lost'
            else:
                infos[self.player_map(1)]['game_result'] = 'tied'
                infos[self.player_map(0)]['game_result'] = 'tied'
        else:
            assert self.curr_time_step.rewards[new_curr_player_id] == 0, "curr_time_step rewards in non terminal state are {}".format(self.curr_time_step.rewards)
            assert self.curr_time_step.rewards[-(new_curr_player_id-1)] == 0

            # dones = {self.player_map(new_curr_player_id): False, "__all__": False}
            rewards = {self.player_map(new_curr_player_id): self.curr_time_step.rewards[new_curr_player_id]}
            assert self.curr_time_step.rewards[1 - new_curr_player_id] == 0.0
            infos = {}

        if self._apply_penalty_for_invalid_actions:
            for player_id, penalty in enumerate(self._invalid_action_penalties):
                if penalty and self.player_map(player_id) in rewards:
                    rewards[self.player_map(player_id)] -= 4.0
                    self._invalid_action_penalties[player_id] = False

        return obs, rewards, dones, infos


register_env(POKER_ENV, lambda env_config: PokerMultiAgentEnv(env_config))
