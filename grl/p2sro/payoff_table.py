import json
from copy import deepcopy
from threading import RLock
from typing import List

import numpy as np

from grl.utils.common import string_to_int_tuple
from grl.utils.strategy_spec import StrategySpec


class PayoffTable(object):

    def __init__(self, n_players: int,
                 exponential_average_coeff=None,
                 restore_strat_ids_to_specs=None,
                 restore_player_and_strat_index_to_strat_ids=None,
                 restore_payoff_matrices_per_player=None,
                 restore_games_played_matrices_per_player=None):

        if n_players < 2:
            raise ValueError("PayoffTables only support 2 or more players.")

        if exponential_average_coeff is not None and not (0 < exponential_average_coeff <= 1):
            raise ValueError("If provided, exponential_average_coeff must be in the range (0, 1]")

        self._n_players = n_players
        self._exponential_average_coeff = exponential_average_coeff

        if restore_strat_ids_to_specs is not None:
            self._strat_ids_to_specs = restore_strat_ids_to_specs
            self._player_and_strat_index_to_strat_id = restore_player_and_strat_index_to_strat_ids
            self._payoff_matrices_per_player = [np.asarray(m, dtype=np.float64) for m in
                                                restore_payoff_matrices_per_player]
            self._games_played_matrices_per_player = [np.asarray(m, dtype=np.int) for m in
                                                      restore_games_played_matrices_per_player]
        else:
            self._player_and_strat_index_to_strat_id = {}
            self._strat_ids_to_specs = {}
            self._payoff_matrices_per_player = [np.ndarray(shape=[0] * n_players, dtype=np.float64) for _ in
                                                range(n_players)]
            self._games_played_matrices_per_player = [np.ndarray(shape=[0] * n_players, dtype=np.int64) for _ in
                                                      range(n_players)]

        self._modification_lock = RLock()

    def copy(self):
        with self._modification_lock:
            return PayoffTable(n_players=self._n_players,
                               exponential_average_coeff=self._exponential_average_coeff,
                               restore_strat_ids_to_specs=deepcopy(self._strat_ids_to_specs),
                               restore_player_and_strat_index_to_strat_ids=deepcopy(
                                   self._player_and_strat_index_to_strat_id),
                               restore_payoff_matrices_per_player=deepcopy(self._payoff_matrices_per_player),
                               restore_games_played_matrices_per_player=deepcopy(
                                   self._games_played_matrices_per_player))

    def _get_json_dict(self):
        with self._modification_lock:
            return {
                "n_players": self._n_players,
                "exponential_average_coeff": self._exponential_average_coeff,
                "strat_ids_to_specs": {strat_id: spec.serialize_to_dict() for strat_id, spec in
                                       self._strat_ids_to_specs.items()},
                "player_and_strat_index_to_strat_id": {str(k): v for k, v in
                                                       self._player_and_strat_index_to_strat_id.items()},
                "payoff_matrices_per_player": [m.tolist() for m in self._payoff_matrices_per_player],
                "games_played_matrices_per_player": [m.tolist() for m in self._games_played_matrices_per_player]
            }

    def to_json_string(self):
        return json.dumps(obj=self._get_json_dict())

    def to_json_file(self, file_path):
        with open(file=file_path, mode="w+") as file:
            json.dump(obj=self._get_json_dict(), fp=file)

    @classmethod
    def from_json_string(cls, json_string):
        json_dict = json.loads(s=json_string)
        strat_ids_to_specs = {
            strat_id: StrategySpec.from_dict(serialized_dict=spec_dict)
            for strat_id, spec_dict in json_dict["strat_ids_to_specs"].items()
        }
        player_and_strat_index_to_strat_ids = {
            string_to_int_tuple(s=player_and_strat_index): strat_id
            for player_and_strat_index, strat_id in json_dict["player_and_strat_index_to_strat_id"].items()
        }
        return PayoffTable(n_players=json_dict["n_players"],
                           exponential_average_coeff=json_dict["exponential_average_coeff"],
                           restore_strat_ids_to_specs=strat_ids_to_specs,
                           restore_player_and_strat_index_to_strat_ids=player_and_strat_index_to_strat_ids,
                           restore_payoff_matrices_per_player=json_dict["payoff_matrices_per_player"],
                           restore_games_played_matrices_per_player=json_dict["games_played_matrices_per_player"])

    @classmethod
    def from_json_file(cls, json_file_path):
        with open(file=json_file_path, mode="r") as file:
            return PayoffTable.from_json_string(json_string=file.read())

    def shape(self):
        with self._modification_lock:
            return self._payoff_matrices_per_player[0].shape

    def size(self):
        with self._modification_lock:
            return self._payoff_matrices_per_player[0].size

    def n_players(self):
        with self._modification_lock:
            return self._n_players

    def get_payoff_matrix_for_player(self, player) -> np.ndarray:
        with self._modification_lock:
            return self._payoff_matrices_per_player[player].copy()

    def get_spec_for_player_and_pure_strat_index(self, player, pure_strat_index) -> StrategySpec:
        with self._modification_lock:
            strat_id = self._player_and_strat_index_to_strat_id[(player, pure_strat_index)]
            return self.get_spec_for_strat_id(strat_id=strat_id)

    def is_strat_id_in_payoff_matrix(self, player, pure_strategy_id):
        with self._modification_lock:
            try:
                spec: StrategySpec = self._strat_ids_to_specs[pure_strategy_id]
                return player in spec.get_pure_strat_indexes().keys()
            except KeyError:
                return False

    def get_spec_for_strat_id(self, strat_id):
        with self._modification_lock:
            try:
                return self._strat_ids_to_specs[strat_id]
            except KeyError:
                return None

    def get_strategy_index_for_player_and_strat_id(self, player, strat_id):
        with self._modification_lock:
            spec: StrategySpec = self.get_spec_for_strat_id(strat_id=strat_id)
            strategy_index = spec.get_pure_strat_indexes().get(player)
            return strategy_index

    def get_ordered_spec_list_for_player(self, player: int) -> List[StrategySpec]:
        with self._modification_lock:
            if player < 0 or player >= self._n_players:
                raise ValueError(
                    f"invalid player index {player}. Valid player indexes are in [0, {self._n_players - 1}].")
            num_policies_for_player = self._payoff_matrices_per_player[0].shape[player]
            spec_list = []
            for strat_index in range(num_policies_for_player):
                spec_list.append(self.get_spec_for_player_and_pure_strat_index(player=player,
                                                                               pure_strat_index=strat_index))
            return spec_list

    def add_new_pure_strategy(self, player, strategy_id, metadata=None) -> StrategySpec:
        with self._modification_lock:
            existing_spec: StrategySpec = self._strat_ids_to_specs.get(strategy_id)
            if existing_spec is not None:
                existing_spec.update_metadata(new_metadata=metadata)
                spec = existing_spec
            else:
                spec = StrategySpec(strategy_id=strategy_id, metadata=metadata)

            new_strategy_index = self._payoff_matrices_per_player[0].shape[player]
            spec.assign_pure_strat_index(player=player, pure_strat_index=new_strategy_index)

            pad_size = 1
            pad_axis = player
            npad = [(0, 0)] * self._payoff_matrices_per_player[0].ndim
            npad[pad_axis] = (0, pad_size)

            for i in range(self._n_players):
                self._payoff_matrices_per_player[i] = np.pad(self._payoff_matrices_per_player[i],
                                                             pad_width=npad, mode='constant', constant_values=0)
                self._games_played_matrices_per_player[i] = np.pad(self._games_played_matrices_per_player[i],
                                                                   pad_width=npad, mode='constant', constant_values=0)

            self._player_and_strat_index_to_strat_id[(player, new_strategy_index)] = spec.id
            self._strat_ids_to_specs[strategy_id] = spec
            return spec

    def add_empirical_payoff_result(self,
                                    as_player: int,
                                    strat_ids_for_each_player,
                                    payoff: float,
                                    games_played: int,
                                    override_all_previous_results: bool):
        with self._modification_lock:
            # Verify all players are included in the payoff result and id's are valid.
            if as_player < 0 or as_player >= self._n_players:
                raise ValueError(f"as_player is an out of range value for this game.")
            if len(strat_ids_for_each_player) != self._n_players:
                raise ValueError(f"A different number of strat ids ({len(strat_ids_for_each_player)}) was provided "
                                 f"than there are players in the game ({self._n_players}).")
            for player, strat_id_to_check in enumerate(strat_ids_for_each_player):
                spec_to_check: StrategySpec = self.get_spec_for_strat_id(strat_id=strat_id_to_check)
                if spec_to_check is None:
                    raise ValueError(f"Strategy id {strat_id_to_check} doesn't exist in the payoff table.")
                if player not in spec_to_check.get_pure_strat_indexes().keys():
                    raise ValueError(f"Strategy id {strat_id_to_check} is in the payoff table, but it isn't "
                                     f"assigned to player {player}.\n"
                                     f"To give the same strategy to multiple players, "
                                     f"add it to the payoff table multiple times, once per player.")
            if games_played < 1:
                raise ValueError("Can't add an empirical payoff result without a "
                                 "positive non-zero amount of games played.")

            strat_indexes_per_player = []
            for player, strat_id in enumerate(strat_ids_for_each_player):
                strat_index_for_player = self.get_strategy_index_for_player_and_strat_id(player=player,
                                                                                         strat_id=strat_id)
                strat_indexes_per_player.append([strat_index_for_player])

            if override_all_previous_results:
                new_games_played = games_played
                new_payoff = payoff
            else:
                old_games_played = self._games_played_matrices_per_player[as_player][tuple(strat_indexes_per_player)]
                old_payoff = self._payoff_matrices_per_player[as_player][tuple(strat_indexes_per_player)]

                new_games_played = games_played + old_games_played

                if self._exponential_average_coeff is None:
                    # new payoff result is an unwieghted average over all results new and previous
                    new_payoff = payoff * (games_played / new_games_played) + old_payoff * (
                                old_games_played / new_games_played)
                else:
                    # new payoffs are weighted by the exponential average coeff times games played
                    new_payoff_weight = min(1.0, self._exponential_average_coeff * games_played)
                    new_payoff = payoff * new_payoff_weight + old_payoff * (1 - new_payoff_weight)

            self._games_played_matrices_per_player[as_player][tuple(strat_indexes_per_player)] = new_games_played
            self._payoff_matrices_per_player[as_player][tuple(strat_indexes_per_player)] = new_payoff
