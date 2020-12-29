import os
import json

from grl.p2sro.p2sro_manager import P2SROManagerLogger, P2SROManager
from grl.p2sro.payoff_table import PayoffTable, PayoffTableStrategySpec
from grl.utils import ensure_dir


class PokerP2SROManagerLogger(P2SROManagerLogger):

    def __init__(self, p2sro_manger: P2SROManager, log_dir: str):
        super().__init__(p2sro_manger, log_dir)
        self._log_dir = log_dir
        self._manager = p2sro_manger

        self._payoff_table_checkpoint_dir = os.path.join(self._log_dir, "payoff_table_checkpoints")
        self._payoff_table_checkpoint_count = 0

    def on_active_policy_moved_to_fixed(self, player: int, policy_num: int, fixed_policy_spec: PayoffTableStrategySpec):
        data = self._manager.get_copy_of_latest_data()
        latest_payoff_table, active_policy_nums_per_player, fixed_policy_nums_per_player = data
        pt_checkpoint_path = os.path.join(self._payoff_table_checkpoint_dir,
                                          f"payoff_table_checkpoint_{self._payoff_table_checkpoint_count}.json")
        policy_nums_path = os.path.join(self._payoff_table_checkpoint_dir,
                                        f"policy_nums_checkpoint_{self._payoff_table_checkpoint_count}.json")
        ensure_dir(file_path=pt_checkpoint_path)
        ensure_dir(file_path=policy_nums_path)

        latest_payoff_table.to_json_file(file_path=pt_checkpoint_path)

        player_policy_nums = {}
        for player_i, (active_policy_nums, fixed_policy_nums) in enumerate(
                zip(active_policy_nums_per_player, fixed_policy_nums_per_player)):
            player_policy_nums[player_i] = {
                "active_policies": active_policy_nums,
                "fixed_policies": fixed_policy_nums
            }

        with open(policy_nums_path, "w+") as policy_nums_file:
            json.dump(obj=player_policy_nums, fp=policy_nums_file)

        self._payoff_table_checkpoint_count += 1
