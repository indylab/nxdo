from typing import Dict

import numpy as np
import random

import gym
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from multiagent.environment import MultiAgentEnv as ParticleMultiAgentEnv
import multiagent.scenarios as scenarios

SYMMETRIC_PUSH_ENV = "symmetric_push"


def _particle_obs_to_dict_obs(particle_obs: np.ndarray, player_map) -> Dict[int, np.ndarray]:
    return {player_map(0): particle_obs[0], player_map(1): particle_obs[1]}

def _dict_action_to_particle_action(dict_action: Dict[int, np.ndarray], player_map) -> np.ndarray:
    agent_actions = [0, 0]
    for agent, action in dict_action.items():
        agent_actions[player_map(agent)] = action
    return np.asarray(agent_actions, dtype=np.int32)

class SimplePushMultiAgentEnv(MultiAgentEnv):

    def __init__(self, env_config=None):
        self._fixed_players = env_config.get("fixed_players", False)  # symmetric game if False
        self._continuous_actions = env_config.get("continuous_actions", False)

        scenario = scenarios.load("zero_sum_simple_push.py").Scenario()
        world = scenario.make_world()
        self.base_env = ParticleMultiAgentEnv(world=world,
                                              reset_callback=scenario.reset_world,
                                              reward_callback=scenario.reward,
                                              observation_callback=scenario.observation,
                                              info_callback=None,
                                              shared_viewer=False,
                                              discrete_action_space=not self._continuous_actions,
                                              discrete_action_input=not self._continuous_actions)

        assert self.base_env.action_space[0] == self.base_env.action_space[1]
        assert self.base_env.observation_space[0] == self.base_env.observation_space[1]

        self.action_space = self.base_env.action_space[0]
        self.observation_space = self.base_env.observation_space[0]

        self._time_limit = 60
        self._episode_step_counter = 0
        self.player_map = None

    def reset(self):
        """Resets the env and returns observations from ready agents.

        Returns:
            obs (dict): New observations for each ready agent.
        """

        # swap player mapping in half of the games
        # assuming only two players
        if self._fixed_players:
            self.player_map = lambda p: p
        else:
            self.player_map = random.choice((lambda p: p,
                                             lambda p: (1 - p)))

        particle_obs = self.base_env.reset()
        self._episode_step_counter = 0
        return _particle_obs_to_dict_obs(particle_obs=particle_obs, player_map=self.player_map)

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
        if len(action_dict) != 2:
            raise ValueError("Both players must make take an action at every step.")

        actions = _dict_action_to_particle_action(dict_action=action_dict, player_map=self.player_map)

        observation, reward, done, info = self.base_env.step(actions)

        assert len(observation) == 2
        assert len(reward) == 2
        assert len(done) == 2

        obs = {self.player_map(player): player_obs for player, player_obs in enumerate(observation)}
        rewards = {self.player_map(player): player_rew for player, player_rew in enumerate(reward)}

        assert rewards[0] == -rewards[1]

        self._episode_step_counter += 1
        if self._episode_step_counter >= self._time_limit:
            dones = {0: True, 1: True, "__all__": True}
        else:
            dones = {0: False, 1: False, "__all__": False}

        infos = {0: {}, 1: {}}

        # self.render()
        #
        # if dones["__all__"]:
        #     print(f"infos: {infos}")
        #     print(f"\n\nrewards {rewards}")
        #     assert False
        return obs, rewards, dones, infos

    def render(self):
        self.base_env.render(mode='human')


register_env(SYMMETRIC_PUSH_ENV, lambda env_config: SimplePushMultiAgentEnv(env_config))
