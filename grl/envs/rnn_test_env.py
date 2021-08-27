import numpy as np
from gym.spaces import Box

from ray.rllib.env.multi_agent_env import MultiAgentEnv

import copy


def with_base_config(base_config, extra_config):
    config = copy.deepcopy(base_config)
    config.update(extra_config)
    return config


class RNNTestMultiAgentEnv(MultiAgentEnv):

    def __init__(self, env_config=None):
        if env_config is None:
            env_config = dict()

        env_config = with_base_config(base_config={
            "obs_shape": (5, 5, 3),
            "agents": [0]
        }, extra_config=env_config)

        self._obs_shape = env_config["obs_shape"]
        self._agents = env_config["agents"]

        self.observation_space = Box(low=0.0, high=1.0, shape=self._obs_shape)
        self.action_space = Box(low=0.0, high=1.0, shape=(1,))

        self._episode_step_counter = 0
        self._target_action = None

    def reset(self):
        self._episode_step_counter = 0
        self._target_action = np.random.random()
        return {a: np.full(shape=self._obs_shape, fill_value=self._target_action, dtype=np.float32)
                for a in self._agents}

    def step(self, action_dict):
        self._episode_step_counter += 1

        obs = {a: np.zeros(shape=self._obs_shape, dtype=np.float32) for a in self._agents}

        if self._episode_step_counter >= 10:
            rews = {a: -abs(action_dict[a] - self._target_action) for a in self._agents}
            dones = {a: True for a in self._agents}
            dones["__all__"] = True
        else:
            rews = {a: 0.0 for a in self._agents}
            dones = {a: False for a in self._agents}
            dones["__all__"] = False

        return obs, rews, dones, {}
