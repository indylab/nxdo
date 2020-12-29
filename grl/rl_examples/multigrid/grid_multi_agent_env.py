import numpy as np

import gym
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from gym.wrappers import TimeLimit
from gym.spaces import Space
# register multigrid gym envs
import social_rl.gym_multigrid
from social_rl.gym_multigrid.envs.adversarial import AdversarialEnv, ReparameterizedAdversarialEnv

MULTIGRID_ENV = "multigrid_env"


class GridMultiAgentEnv(MultiAgentEnv):

    def __init__(self, env_config=None):
        env_version = env_config['version']
        gym_spec = gym.spec(env_version)
        gym_env = gym_spec.make({})

        self._is_adversarial = isinstance(gym_env, AdversarialEnv) or isinstance(gym_env, ReparameterizedAdversarialEnv)
        self._is_adversary_phase_in_episode = True
        self._is_antagonist_phase_in_episode = False

        self._env = TimeLimit(gym_env)

        # Good discussion of how action_space and observation_space work in MultiAgentEnv:
        # https://github.com/ray-project/ray/issues/6875
        # These properties below are not part of the MultiAgentEnv parent class.
        # RLLib doesn't observe these member variables in a multi agent env.
        # We just need to pass in the correct obs/action spaces to the configs for each policy.
        self.action_space = {
            "adversary": self._env.adversary_action_space,
            "protagonist": self._env.action_space,
            "antagonist": self._env.action_space,
        }
        self.observation_space = {
            "adversary": self._env.adversary_observation_space,
            "protagonist": self._env.observation_space,
            "antagonist": self._env.observation_space,
        }

    def reset(self):
        """Resets the env and returns observations from ready agents.

        Returns:
            obs (dict): New observations for each ready agent.
        """
        observation = self._env.reset()

        if self._is_adversarial:
            self._is_adversary_phase_in_episode = True
            self._is_antagonist_phase_in_episode = False
            assert len(observation) == 3 or len(observation) == 5
            return {
                "adversary": observation
            }

        # Single player game
        assert len(observation) == 2
        return {
            "protagonist": observation
        }

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

        obs = {}
        rews = {}
        dones = {}
        infos = {}

        # TODO: run adversary phase, then reset agent and run as antagonist, then reset agent and run as protagonist.
        # TODO: Then calculate rewards

        if self._is_adversarial and self._is_adversary_phase_in_episode:
            adversary_action = action_dict["adversary"]
            if not adversary_action in self.action_space["adversary"]:
                raise ValueError(f"Adversary action {adversary_action} is not in adversary action space "
                                 f"{self.action_space['adversary']}")
            adv_obs, adv_rew, adv_done, adv_info = self._env.step_adversary(adversary_action)
            obs["adversary"] = adv_obs
            rews["adversary"] = adv_rew
            dones["adversary"] = adv_done
            infos["adversary"] = adv_info
        else:
            protagonist_action = action_dict["protagonist"]
            if not protagonist_action in self.action_space["protagonist"]:
                raise ValueError(f"Protagonist action {protagonist_action} is not in protagonist action space "
                                 f"{self.action_space['protagonist']}")
            pro_obs, pro_rew, pro_done, pro_info = self._env.step(protagonist_action)
            obs["protagonist"] = pro_obs
            rews["protagonist"] = pro_rew
            dones["protagonist"] = pro_done
            infos["protagonist"] = pro_info

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
            rewards = {player: player_info["main_reward"]/2000.0 for player, player_info in enumerate(info)}
        dones = {0: True, 1: True, "__all__": True} if done[0] else {0: False, 1: False, "__all__": False}
        infos = {player: player_info for player, player_info in enumerate(info)}

        return obs, rewards, dones, infos

    def render(self):
        self._env.render(mode='human')


register_env(MULTIGRID_ENV, lambda env_config: GridMultiAgentEnv(env_config))
