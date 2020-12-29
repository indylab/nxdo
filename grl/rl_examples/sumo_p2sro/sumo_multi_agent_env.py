import numpy as np

import gym
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from robosumo.envs import SumoEnv

SUMO_ENV = "sumo_env"

# from 10,000 observations with random policies
_RANDOM_POLICY_OBS_MEANS = np.asarray([-0.059464777201391435, 0.0367132189638141, 1.1212695160922874, 0.5916771742523922, -0.018387594505578326, -0.07473146513074544, -8.180347961936533e-05, 0.3367031205434438, -0.8685671601925177, -0.103245943722769, 0.9724212114258398, -0.19052258668268557, -0.8201792915269491, 0.004580059733674719, 0.9975300472788869, 0.002608828709141751, 0.04884006269981829, -0.05019080849792082, -0.08413472654621432, -0.043263522615618756, -0.01905409995732477, 0.05369180740225651, -0.2282620101178078, -0.00017209129127553275, 0.22062440400628303, -0.05875292400837649, -0.25510266240594975, 0.017943824684392925, 0.24972723828051846, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.05946477720139143, 0.03671321896381409, 1.1212695160922874, 0.5916771742523921, -0.018387594505578333, -0.07473146513074541, -8.180347961936431e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

_RANDOM_POLICY_OBS_STDS = np.asarray([0.959792509305965, 0.935868723894452, 0.25366209680005625, 0.4551474590040157, 0.387990902924668, 0.4112056303181713, 0.3423579978167994, 0.3144137486372087, 0.32360597390125373, 0.42517366423399977, 0.30668728120581096, 0.4096663509885993, 0.31490804958150287, 0.4225740750218721, 0.3124177815778859, 0.8086277194274057, 0.809612019585763, 0.9218572692985189, 2.0850945169737582, 1.7693466054152152, 2.0685912555924584, 4.173454810053246, 5.019033485858277, 5.59286717779848, 5.069661786455332, 5.1783046323032895, 4.628867688082865, 5.788879823572316, 4.670887375967568, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.959792509305965, 0.935868723894452, 0.25366209680005625, 0.4551474590040157, 0.387990902924668, 0.41120563031817126, 0.3423579978167994, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
_RANDOM_POLICY_OBS_STDS[_RANDOM_POLICY_OBS_STDS == 0.0] = 1


def _normalize_obs(obs: np.ndarray):
    return np.clip(a=(obs - _RANDOM_POLICY_OBS_MEANS) / _RANDOM_POLICY_OBS_STDS, a_min=-10, a_max=10)


class SumoMultiAgentEnv(MultiAgentEnv):

    def __init__(self, env_config=None):
        self._env = gym.make("RoboSumo-Ant-vs-Ant-v0")

        self._use_shaping_rewards = env_config.get("shaping_rewards", False)
        self._scale_rewards_coeff = env_config.get("scale_rewards", 1.0)

        assert self._env.action_space.spaces[0] == self._env.action_space.spaces[1]
        assert self._env.observation_space.spaces[0] == self._env.observation_space.spaces[1]

        self.action_space = self._env.action_space.spaces[0]
        self.observation_space = self._env.observation_space.spaces[0]

        self._total_shaping_rews = {player: 0.0 for player in range(2)}
        self._total_main_rews = {player: 0.0 for player in range(2)}

    def reset(self):
        """Resets the env and returns observations from ready agents.

        Returns:
            obs (dict): New observations for each ready agent.
        """
        self._total_shaping_rews = {player: 0.0 for player in range(2)}
        self._total_main_rews = {player: 0.0 for player in range(2)}
        observation = self._env.reset()
        assert len(observation) == 2
        obs = {player: _normalize_obs(player_obs) for player, player_obs in enumerate(observation)}
        return obs

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
            raise ValueError("Both players must make take an action at every step for SumoMultiAgentEnv")

        actions = [None, None]

        for player, action in action_dict.items():
            actions[player] = action

        observation, reward, done, info = self._env.step(tuple(actions))

        assert len(observation) == 2
        assert len(reward) == 2
        assert len(done) == 2
        assert len(info) == 2

        obs = {player: _normalize_obs(obs=player_obs) for player, player_obs in enumerate(observation)}
        if done[0] and 'winner' not in info[0] and 'winner' not in info[1]:
            # draw
            rewards = {player: 0 for player, player_info in enumerate(info)}
        else:
            rewards = {}
            for player, player_info in enumerate(info):
                main_reward = player_info["main_reward"] * self._scale_rewards_coeff
                self._total_main_rews[player] += main_reward
                if self._use_shaping_rewards:
                    shaping_reward = player_info["shaping_reward"] * self._scale_rewards_coeff
                else:
                    shaping_reward = 0.0
                self._total_shaping_rews[player] += shaping_reward

                rewards[player] = main_reward + shaping_reward

        for player, player_info in enumerate(info):
            player_info["total_main_reward"] = self._total_main_rews[player]
            player_info["total_shaping_reward"] = self._total_shaping_rews[player]
            if done[0]:
                player_info["lost"] = player_info["main_reward"] < 0

        dones = {0: True, 1: True, "__all__": True} if done[0] else {0: False, 1: False, "__all__": False}
        infos = {player: player_info for player, player_info in enumerate(info)}
        #
        # if dones["__all__"]:
        #     print(f"infos: {infos}")
        #     print(f"\n\nrewards {rewards}")
        #     assert False
        return obs, rewards, dones, infos

    def render(self):
        self._env.render(mode='human')


register_env(SUMO_ENV, lambda env_config: SumoMultiAgentEnv(env_config))
