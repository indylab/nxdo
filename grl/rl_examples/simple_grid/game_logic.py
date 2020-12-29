from typing import Tuple
import numpy as np

_Empty = 0.0
_Player = 0.5  # also denotes spawn in adversary observation
_Goal = 1.0
_Lava = -0.5
_Goal_In_Lava = -1.0

_BASIC_SMALL_LAYOUT = np.asarray([
    [_Empty, _Empty, _Empty],
    [_Empty, _Empty, _Empty],
    [_Empty, _Empty, _Empty],
], dtype=np.int32)

_BASIC_MEDIUM_LAYOUT = np.asarray([
    [_Empty, _Empty, _Empty, _Empty, _Empty],
    [_Empty, _Empty, _Empty, _Empty, _Empty],
    [_Empty, _Empty, _Empty, _Empty, _Empty],
    [_Empty, _Empty, _Empty, _Empty, _Empty],
    [_Empty, _Empty, _Empty, _Empty, _Empty],
], dtype=np.int32)

_LAVA_LAYOUT = np.asarray([
    [_Lava, _Lava,  _Lava,  _Lava,  _Lava],
    [_Lava, _Empty, _Empty, _Empty, _Lava],
    [_Lava, _Empty, _Empty, _Empty, _Lava],
    [_Lava, _Empty, _Empty, _Empty, _Lava],
    [_Lava, _Lava,  _Lava,  _Lava,  _Lava],
], dtype=np.int32)


class SimpleGridGame:

    def __init__(self, layout_preset_name="basic_small", goal_can_be_in_lava=False, protagonist_sees_goal=False):

        self._layout_preset_name = layout_preset_name
        self.goal_can_be_in_lava = goal_can_be_in_lava
        self.protagonist_sees_goal = protagonist_sees_goal

        self.player_location_xy = None
        self.spawn_location_xy = None
        self.goal_location_xy = None
        self.is_setup_phase = True
        self.is_done = False
        self.player_reached_goal = False

        self.reset_all()
        self.rows, self.columns = self.state_layout.shape

        # see whole level
        # (adversary sees layout with player spawn marked if set)
        # (protagonist sees layout with player location marked)
        self.observation_space_1d_len = len(self.state_layout.flatten())

        # pick a place to put an object (pick spawn, then goal)
        self.adversary_action_space_1d_len = len(self.state_layout.flatten())

        # move in 4 directions
        self.protagonist_action_space_len = 4

    def reset_player(self):
        # resets player location while keeping the same layout currently set
        if self.is_setup_phase:
            raise ValueError("Can't reset player during setup phase. There's no player present to reset.")
        assert self.spawn_location_xy is not None
        self.player_location_xy = self.spawn_location_xy.copy()
        self.is_done = False
        self.player_reached_goal = False

    def reset_all(self):
        # resets layout and begins setup phase anew
        self.player_location_xy = None
        self.spawn_location_xy = None
        self.goal_location_xy = None
        self._set_state_layout_from_preset(layout_preset_name=self._layout_preset_name)
        self.is_setup_phase = True
        self.is_done = False
        self.player_reached_goal = False

    def _set_state_layout_from_preset(self, layout_preset_name: str):
        if layout_preset_name == "basic_small":
            self.state_layout: np.ndarray = _BASIC_SMALL_LAYOUT.copy()
        elif layout_preset_name == "basic_medium":
            self.state_layout: np.ndarray = _BASIC_MEDIUM_LAYOUT.copy()
        elif layout_preset_name == "lava":
            self.state_layout: np.ndarray = _LAVA_LAYOUT.copy()
        else:
            raise NotImplementedError(f"Unknown layout preset: {layout_preset_name}")

    def get_1d_index_from_2d_location(self, xy: np.ndarray) -> int:
        # row major indexing
        return self.rows * xy[0] + xy[1]

    def get_2d_location_from_1d_index(self, index: int) -> np.ndarray:
        # row major indexing
        return np.asarray([index // self.rows, index % self.rows])

    def is_cell_valid_spawn_location(self, xy: np.ndarray) -> bool:
        # spawn location has to be in an empty cell and can't be the same as the goal location
        if self.state_layout[xy[0], xy[1]] != _Empty:
            return False
        if self.goal_location_xy is not None and np.array_equal(self.goal_location_xy, xy):
            return False
        return True

    def is_cell_valid_goal_location(self, xy: np.ndarray) -> bool:
        # goal location has to be empty (or lava if allowed) and can't be the same as the spawn location
        ok_layout_cell_types = [_Empty, _Lava] if self.goal_can_be_in_lava else [_Empty]
        if self.state_layout[xy[0], xy[1]] not in ok_layout_cell_types:
            return False
        if self.spawn_location_xy is not None and np.array_equal(self.spawn_location_xy, xy):
            return False
        return True

    def is_cell_valid_lava_location(self, xy: np.ndarray) -> bool:
        # lava location has to be empty and can't be the same as the spawn location,
        # can be same as goal if allowed
        if self.state_layout[xy[0], xy[1]] != _Empty:
            return False
        if self.spawn_location_xy is not None and np.array_equal(self.spawn_location_xy, xy):
            return False
        if not self.goal_can_be_in_lava and \
                self.goal_location_xy is not None and \
                np.array_equal(self.goal_location_xy, xy):
            return False
        return True

    def set_spawn(self, index: int) -> bool:
        # returns whether spawn was successfully set
        if not self.is_setup_phase:
            return False
        spawn_xy = self.get_2d_location_from_1d_index(index=index)
        if not self.is_cell_valid_spawn_location(xy=spawn_xy):
            return False
        self.spawn_location_xy = spawn_xy
        return True

    def set_goal(self, index: int) -> bool:
        # returns whether goal was successfully set
        if not self.is_setup_phase:
            return False
        goal_xy = self.get_2d_location_from_1d_index(index=index)
        if not self.is_cell_valid_goal_location(xy=goal_xy):
            return False
        self.goal_location_xy = goal_xy
        return True

    def set_lava(self, index: int) -> bool:
        # returns whether lava was successfully set
        if not self.is_setup_phase:
            return False
        lava_xy = self.get_2d_location_from_1d_index(index=index)
        if not self.is_cell_valid_lava_location(xy=lava_xy):
            return False
        self.state_layout[lava_xy[0], lava_xy[1]] = _Lava
        return True

    def end_setup_phase(self):
        if self.spawn_location_xy is None:
            raise ValueError("Spawn location isn't set yet.")
        if self.goal_location_xy is None:
            raise ValueError("Goal location isn't set yet.")
        if self.is_setup_phase:
            raise ValueError("Already in setup phase.")
        self.is_setup_phase = False
        self.reset_player()

    def get_adversary_observation(self) -> np.ndarray:
        adversary_obs: np.ndarray = self.state_layout.copy()
        if self.spawn_location_xy is not None:
            adversary_obs[self.spawn_location_xy[0], self.spawn_location_xy[1]] = _Player
        if self.goal_location_xy is not None:
            goal_value = _Goal_In_Lava if self.goal_can_be_in_lava else _Goal
            adversary_obs[self.goal_location_xy[0], self.goal_location_xy[1]] = goal_value
        return adversary_obs.flatten()

    def set_protagonist_goal_visibility(self, can_see_goal: bool):
        self.protagonist_sees_goal = can_see_goal

    def get_protagonist_observation(self) -> np.ndarray:
        protagonist_obs: np.ndarray = self.state_layout.copy()
        protagonist_obs[self.player_location_xy[0], self.player_location_xy[1]] = _Player
        if self.protagonist_sees_goal:
            goal_value = _Goal_In_Lava if self.goal_can_be_in_lava else _Goal
            protagonist_obs[self.goal_location_xy[0], self.goal_location_xy[1]] = goal_value
        return protagonist_obs.flatten()

    @staticmethod
    def _action_to_direction_vector(action: int) -> np.ndarray:
        if action == 0:
            direction_xy = [0, 1]
        elif action == 1:
            direction_xy = [0, -1]
        elif action == 2:
            direction_xy = [1, 0]
        elif action == 3:
            direction_xy = [-1, 0]
        else:
            raise ValueError(f"Out of range action (should be 0|1|2|3): {action}")
        return np.asarray(direction_xy, dtype=np.int32)

    def move_player(self, action: int) -> Tuple[bool, bool]:
        if self.is_setup_phase:
            raise ValueError("Can't move player during setup phase. There's no player present to move.")
        if self.is_done:
            raise ValueError("Player is done and can't move any more.")
        assert self.player_reached_goal is False

        direction_vector_xy = self._action_to_direction_vector(action=action)

        new_player_location_xy = self.player_location_xy + direction_vector_xy

        can_move_to_destination = True
        destination_cell_type = None
        try:
            destination_cell_type = self.state_layout[new_player_location_xy[0], new_player_location_xy[1]]
        except IndexError:
            can_move_to_destination = False
        if can_move_to_destination:
            if destination_cell_type == _Empty:
                self.player_location_xy = new_player_location_xy
            elif destination_cell_type in [_Lava, _Goal_In_Lava]:
                # Going to goal-in-lava does not mean player reaches goal. Player "dies" instead.
                self.is_done = True
            elif destination_cell_type == _Goal:
                self.is_done = True
                self.player_reached_goal = True

        return self.is_done, self.player_reached_goal
