from abc import abstractmethod

from ray.rllib.env import MultiAgentEnv


class ValidActionsMultiAgentEnv(MultiAgentEnv):

    @property
    @abstractmethod
    def observation_length(self):
        pass

    @observation_length.setter
    @abstractmethod
    def observation_length(self, value):
        pass
