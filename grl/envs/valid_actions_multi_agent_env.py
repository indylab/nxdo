from abc import abstractmethod

from ray.rllib.env import MultiAgentEnv


class ValidActionsMultiAgentEnv(MultiAgentEnv):

    @abstractmethod
    def __init__(self):
        super(ValidActionsMultiAgentEnv, self).__init__()
        self.observation_length = None
        self.orig_observation_length = None
