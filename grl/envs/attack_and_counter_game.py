import numpy as np
from gym.spaces import Box, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils import merge_dicts

MAX_HEALTH = 10
MAX_COUNTERS = 5
MAX_ATTACK_AMOUNT = 2


DEFAULT_ATTACK_COUNTER_GAME_CONFIG = {
    "discrete_actions_for_players": [],
    "discrete_action_space_is_default": False,
}


class AttackCounterGameMultiAgentEnv(MultiAgentEnv):

    def __init__(self, env_config):
        config = merge_dicts(DEFAULT_ATTACK_COUNTER_GAME_CONFIG, env_config)

        self.continuous_action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.discrete_action_space = Discrete(n=501)
        self.discrete_actions_for_players = config["discrete_actions_for_players"]

        for p in self.discrete_actions_for_players:
            if p not in [0, 1]:
                raise ValueError("Values for discrete_actions_for_players can only be [], [0], [1], or [0, 1]")
        if len(self.discrete_actions_for_players) == 2 or config["discrete_action_space_is_default"]:

            self.action_space = self.discrete_action_space

        else:
            self.action_space = self.continuous_action_space

        self.high = self.continuous_action_space.high
        self.low = self.continuous_action_space.low

        self.observation_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self.player1_health = MAX_HEALTH
        self.player1_num_counters = MAX_COUNTERS

        self.player2_health = MAX_HEALTH
        self.player2_num_counters = MAX_COUNTERS

        self.player1 = 0
        self.player2 = 1

    @staticmethod
    def normalize_health(val):
        normalized = (2*val / MAX_HEALTH)-1
        return normalized

    @staticmethod
    def normalize_counters(val):
        normalized = (2*val / MAX_COUNTERS)-1
        return normalized

    @staticmethod
    def parse_action(action):
        # action is [attack, counter]

        attack_percent = (action[0]+1)/2
        assert (attack_percent >= 0 and attack_percent <= 1)
        attack_amount = attack_percent * MAX_ATTACK_AMOUNT

        counter = True if action[1] > action[0] else False

        return attack_amount, counter

    def attack_and_counter_game(self, move1, move2):

        attack1, counter1 = self.parse_action(move1)
        attack2, counter2 = self.parse_action(move2)

        if counter1:
            attack1 = 0
            if self.player1_num_counters == 0:
                counter1 = False
            else:
                self.player1_num_counters -= 1

        if counter2:
            attack2 = 0
            if self.player2_num_counters == 0:
                counter2 = False
            else:
                self.player2_num_counters -= 1

        if counter1:
            attack1 = attack2

        if counter2:
            attack2 = attack1

        self.player1_health = max(0, self.player1_health - attack2)
        self.player2_health = max(0, self.player2_health - attack1)

        return

    def discrete_to_continuous_action(self, discrete_action):
        # discrete_action is the index at which user attacks. convert this to a percentage
        if discrete_action == 500:
            # counter
            continuous_action = np.array([-1, 1], dtype=np.float32)

        else:

            percent_in_range = discrete_action / 499.0
            continuous_action = (percent_in_range * (self.high[0] - self.low[0])) + self.low[0]
            continuous_action = np.array([continuous_action, -1], dtype=np.float32)
            assert 0 - 1e-14 <= percent_in_range <= 1 + 1e-14

            assert continuous_action in self.continuous_action_space, continuous_action
            # This isn't true if self.action_space itself is discrete.

        return continuous_action

    def get_observation(self):

        health1 = self.normalize_health(self.player1_health)
        health2 = self.normalize_health(self.player2_health)
        counter1 = self.normalize_counters(self.player1_num_counters)
        counter2 = self.normalize_counters(self.player2_num_counters)

        return np.asarray([health1, counter1, health2, counter2], dtype=np.float32)

    def reset(self):

        self.player1_health = MAX_HEALTH
        self.player1_num_counters = MAX_COUNTERS
        self.player2_health = MAX_HEALTH
        self.player2_num_counters = MAX_COUNTERS

        obs = self.get_observation()

        return {
            self.player1: obs,
            self.player2: obs
        }

    def step(self, action_dict):
        move1 = action_dict[self.player1]
        move2 = action_dict[self.player2]

        if self.player1 in self.discrete_actions_for_players:
            move1 = self.discrete_to_continuous_action(discrete_action=move1)

        if self.player2 in self.discrete_actions_for_players:
            move2 = self.discrete_to_continuous_action(discrete_action=move2)

        assert move1 in self.continuous_action_space, (f"move1: {move1}, move2: {move2}", self.action_space)
        assert move2 in self.continuous_action_space, (f"move1: {move1}, move2: {move2}", self.action_space)
        assert -1 - 1e-10 <= move1[0] <= 1 + 1e-10
        assert -1 - 1e-10 <= move2[0] <= 1 + 1e-10

        self.attack_and_counter_game(move1, move2)
        done = self.player1_health <= 0 or self.player2_health <= 0

        obs_arr = self.get_observation()

        obs = {
            self.player1: obs_arr,
            self.player2: obs_arr,
        }

        if self.player1_health == 0.0 and self.player2_health == 0.0:
            r1 = 0.0
            r2 = 0.0
        elif self.player1_health == 0:
            r1 = -1.0
            r2 = 1.0
        elif self.player2_health == 0:
            r1 = 1.0
            r2 = -1.0
        else:
            r1 = 0.0
            r2 = 0.0

        assert np.isclose(r1 + r2, 0)

        rew = {
            self.player1: r1,
            self.player2: r2,
        }

        dones = {
            self.player1: done,
            self.player2: done,
            "__all__": done,
        }

        return obs, rew, dones, {}