import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Tuple

from grl.utils.common import ensure_dir
from grl.utils.strategy_spec import StrategySpec

logger = logging.getLogger(__name__)


class P2SROManagerLogger(ABC):
    """
    Logging for a P2SROManager.
    Extend this class to add additional functionality like tracking exploitability over time for small games.
    """

    @abstractmethod
    def __init__(self, p2sro_manger, log_dir: str):
        pass

    def on_new_active_policy(self, player: int, new_policy_num: int, new_policy_spec: StrategySpec):
        pass

    def on_new_active_policy_metadata(self, player: int, policy_num: int, new_policy_spec: StrategySpec):
        pass

    def on_active_policy_moved_to_fixed(self, player: int, policy_num: int, fixed_policy_spec: StrategySpec):
        pass

    def on_payoff_result(self,
                         policy_specs_for_each_player: Tuple[StrategySpec],
                         payoffs_for_each_player: Tuple[float],
                         games_played: int,
                         overrode_all_previous_results: bool):
        pass


class SimpleP2SROManagerLogger(P2SROManagerLogger):
    """
    Saves payoff table checkpoints every time an active policy is set to fixed.
    """

    def __init__(self, p2sro_manger, log_dir: str):
        super().__init__(p2sro_manger, log_dir)
        self._log_dir = log_dir
        self._manager = p2sro_manger

        self._payoff_table_checkpoint_dir = os.path.join(self._log_dir, "payoff_table_checkpoints")
        self._payoff_table_checkpoint_count = 0

        self._latest_numbered_payoff_table_checkpoint_path = None
        self._latest_numbered_policy_nums_path = None

    def on_new_active_policy(self, player: int, new_policy_num: int, new_policy_spec: StrategySpec):
        logger.info(f"Player {player} active policy {new_policy_num} claimed")

    def on_new_active_policy_metadata(self, player: int, policy_num: int, new_policy_spec: StrategySpec):
        pass

    def on_active_policy_moved_to_fixed(self, player: int, policy_num: int, fixed_policy_spec: StrategySpec):
        logger.info(f"Player {player} policy {policy_num} moved to fixed.")

        # save a checkpoint of the payoff table
        data = self._manager.get_copy_of_latest_data()
        latest_payoff_table, active_policy_nums_per_player, fixed_policy_nums_per_player = data

        self._latest_numbered_payoff_table_checkpoint_path = os.path.join(self._payoff_table_checkpoint_dir,
                                                   f"payoff_table_checkpoint_{self._payoff_table_checkpoint_count}.json")
        self._latest_numbered_policy_nums_path = os.path.join(self._payoff_table_checkpoint_dir,
                                                 f"policy_nums_checkpoint_{self._payoff_table_checkpoint_count}.json")

        pt_checkpoint_paths = [os.path.join(self._payoff_table_checkpoint_dir, f"payoff_table_checkpoint_latest.json"),
                               self._latest_numbered_payoff_table_checkpoint_path]
        policy_nums_paths = [os.path.join(self._payoff_table_checkpoint_dir, f"policy_nums_checkpoint_latest.json"),
                             self._latest_numbered_policy_nums_path]

        for pt_checkpoint_path, policy_nums_path in zip(pt_checkpoint_paths, policy_nums_paths):
            ensure_dir(file_path=pt_checkpoint_path)
            ensure_dir(file_path=policy_nums_path)

            latest_payoff_table.to_json_file(file_path=pt_checkpoint_path)
            print(f"\n\n\nSaved payoff table checkpoint to {pt_checkpoint_path}")

            player_policy_nums = {}
            for player_i, (active_policy_nums, fixed_policy_nums) in enumerate(
                    zip(active_policy_nums_per_player, fixed_policy_nums_per_player)):
                player_policy_nums[player_i] = {
                    "active_policies": active_policy_nums,
                    "fixed_policies": fixed_policy_nums
                }

            with open(policy_nums_path, "w+") as policy_nums_file:
                json.dump(obj=player_policy_nums, fp=policy_nums_file)
            print(f"Saved policy nums checkpoint to {policy_nums_path}\n\n\n")

        # append checkpoints metadata to checkpoints_manifest.txt
        checkpoints_manifest_path = os.path.join(self._payoff_table_checkpoint_dir, "checkpoints_manifest.json")
        ensure_dir(file_path=checkpoints_manifest_path)
        with open(checkpoints_manifest_path, "a+") as manifest_file:
            if all(len(fixed_policy_nums) > 0 for fixed_policy_nums in fixed_policy_nums_per_player):
                highest_fixed_policies_for_all_players = min(
                    max(fixed_policy_nums) for fixed_policy_nums in fixed_policy_nums_per_player)
            else:
                highest_fixed_policies_for_all_players = None
            manifest_json_line = json.dumps({"payoff_table_checkpoint_num": self._payoff_table_checkpoint_count,
                                             "highest_fixed_policies_for_all_players": highest_fixed_policies_for_all_players,
                                             "payoff_table_json_path": self._latest_numbered_payoff_table_checkpoint_path,
                                             "policy_nums_json_path": self._latest_numbered_policy_nums_path})
            manifest_file.write(f"{manifest_json_line}\n")

        self._payoff_table_checkpoint_count += 1

    def on_payoff_result(self, policy_specs_for_each_player: Tuple[StrategySpec],
                         payoffs_for_each_player: Tuple[float], games_played: int,
                         overrode_all_previous_results: bool):
        pass
        # json_specs = [spec.to_json() for spec in policy_specs_for_each_player]
        # logger.debug(f"Payoff result for {json_specs}, payoffs: {payoffs_for_each_player}, games: {games_played},"
        #             f" overrides existing results: {overrode_all_previous_results}")
        #
        #
        # data = self._manager.get_copy_of_latest_data()
        # latest_payoff_table, active_policy_nums_per_player, fixed_policy_nums_per_player = data
        # latest_payoff_table: PayoffTable = latest_payoff_table

        # print("Player 0 matrix ---------------------------------------")
        # print(latest_payoff_table.get_payoff_matrix_for_player(0))
        # print("------------------------------------------------------")
        # print("Player 1 matrix ---------------------------------------")
        # print(latest_payoff_table.get_payoff_matrix_for_player(1))
        # print("------------------------------------------------------")

    def get_current_checkpoint_num(self):
        return self._payoff_table_checkpoint_count

    def get_latest_numbered_payoff_table_checkpoint_path(self):
        return self._latest_numbered_payoff_table_checkpoint_path

    def get_latest_numbered_policy_nums_path(self):
        return self._latest_numbered_policy_nums_path
