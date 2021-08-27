import math
from math import pi
import numpy as np
from gym.spaces import Box, Discrete
from ray.rllib.utils import merge_dicts
from ray.rllib.env.multi_agent_env import MultiAgentEnv


def direction_to_coord(direction):
    """takes x \in [0,1), returns 2d coords on unit circle"""
    return tuple([math.cos(2*pi*direction), math.sin(2*pi*direction)])


def loss_game(coords1, coords2, dim=2, starting_coords=False):
    if not starting_coords:
        starting_coords = [0 for i in range(dim)]
    coords1 = list(coords1)
    coords2 = list(coords2)
    assert len(coords1) == len(coords2) == len(starting_coords) == dim
    new_coords = [coords1[i] + coords2[i] + starting_coords[i] for i in range(dim)]

    val = math.sin(sum([x for x in new_coords])) + (1 / dim) * sum([math.sin(x) for x in new_coords])

    return val, new_coords


def loss_game_independent_sins(coords1, coords2, dim=2, starting_coords=False):
    if not starting_coords:
        starting_coords = [0 for i in range(dim)]
    coords1 = list(coords1)
    coords2 = list(coords2)
    assert len(coords1) == len(coords2) == len(starting_coords) == dim
    new_coords = [coords1[i] + coords2[i] + starting_coords[i] for i in range(dim)]

    val = sum([math.sin(x) for x in new_coords])

    return val, new_coords


DEFAULT_LOSS_GAME_CONFIG = {
    "total_moves": 10,
    "dim": 5,
    "max_move_amount": 2.0,
    "num_actions_per_dim": 5,
    "use_independent_sins": False,
    "discrete_actions_for_players": [],
}


class LossGameMultiDimMultiAgentEnv(MultiAgentEnv):
    def __init__(self, env_config):
        config = merge_dicts(DEFAULT_LOSS_GAME_CONFIG, env_config)
        self.total_moves = config["total_moves"]
        self.dim = config["dim"]
        self.max_moves_amount = config["max_move_amount"]
        self.discrete_actions_for_players = config["discrete_actions_for_players"]

        self.continuous_action_space = Box(low=-1.0, high=1.0, shape=(self.dim,), dtype=np.float32)

        self.num_discrete_actions_per_dim = config["num_actions_per_dim"]
        self.discrete_action_space = Discrete(n=self.num_discrete_actions_per_dim**self.dim)

        self.use_independent_sins = config["use_independent_sins"]

        for p in self.discrete_actions_for_players:
            if p not in [0, 1]:
                raise ValueError("Values for discrete_actions_for_players can only be [], [0], [1], or [0, 1]")
        if len(self.discrete_actions_for_players) == 2:
            self.action_space = self.discrete_action_space
        else:
            self.action_space = self.continuous_action_space

        self.observation_space = Box(low=-500.0, high=500.0, shape=(self.dim + 1,), dtype=np.float32)
        self.player1 = 0
        self.player2 = 1
        self.num_moves = 0
        self.current_coord = [0 for _ in range(self.dim)]
        self.obs = np.array([0 for _ in range(self.dim + 1)])  # coords + val (val is last dim)

    def _discrete_to_continuous_action(self, discrete_action):

        ndim_indices = np.unravel_index(discrete_action, shape=tuple([self.num_discrete_actions_per_dim for _ in range(self.dim)]))

        assert len(ndim_indices) == self.dim

        ndim_percentages = np.asarray(ndim_indices, dtype=np.float32) / (self.num_discrete_actions_per_dim - 1.0)
        assert all(0 - 1e-10 <= p <= 1 + 1e-10 for p in ndim_percentages)

        continuous_action = (ndim_percentages * (self.continuous_action_space.high - self.continuous_action_space.low)) + self.continuous_action_space.low

        assert len(continuous_action) == self.dim
        assert len(continuous_action.shape) == 1
        assert continuous_action in self.continuous_action_space, f"continuous_action {continuous_action}, discrete_action {discrete_action}"

        return continuous_action

    def reset(self):
        self.num_moves = 0
        self.current_coord = [0 for _ in range(self.dim)]
        self.obs = np.array([0 for _ in range(self.dim + 1)])  # coords + val (val is last dim)
        return {
            self.player1: self.obs,
            self.player2: self.obs,
        }

    def step(self, action_dict):
        move1 = np.asarray(action_dict[self.player1])
        move2 = np.asarray(action_dict[self.player2])

        if self.player1 in self.discrete_actions_for_players:
            move1 = self._discrete_to_continuous_action(discrete_action=move1)

        if self.player2 in self.discrete_actions_for_players:
            move2 = self._discrete_to_continuous_action(discrete_action=move2)

        assert move1 in self.continuous_action_space, f"move1: {move1}, move2: {move2}"
        assert move2 in self.continuous_action_space, f"move1: {move1}, move2: {move2}"

        move1 = move1 * self.max_moves_amount
        move2 = move2 * self.max_moves_amount

        if self.use_independent_sins:
            val, new_coords = loss_game_independent_sins(move1, move2, self.dim, self.current_coord)
        else:
            val, new_coords = loss_game(move1, move2, self.dim, self.current_coord)
        r1 = val
        r2 = -val
        new_obs = new_coords.copy()
        new_obs.append(val)
        self.obs = np.array(new_obs)
        obs = {
            self.player1: self.obs,
            self.player2: self.obs,
        }
        self.current_coord = new_coords
        rew = {
            self.player1: r1,
            self.player2: r2,
        }
        self.num_moves += 1
        done = self.num_moves >= self.total_moves
        dones = {
            self.player1: done,
            self.player2: done,
            "__all__": done,
        }
        return obs, rew, dones, {}