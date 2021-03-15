from itertools import product
from threading import RLock
from typing import List, Tuple

import numpy as np

from grl.algos.p2sro.eval_dispatcher import EvalDispatcherWithServer, EvalResult
from grl.algos.p2sro.p2sro_manager.logger import P2SROManagerLogger, SimpleP2SROManagerLogger
from grl.algos.p2sro.payoff_table import PayoffTable
from grl.utils.common import datetime_str, check_if_jsonable
from grl.utils.strategy_spec import StrategySpec


class _P2SROPlayerStats(object):
    """
    Keeps track of active and fixed policy nums for a given player in a game.
    """

    def __init__(self, player):
        self.player = player
        self.fixed_policy_indexes = []
        self.active_policy_indexes = []
        self._next_active_policy = 0

    def get_new_active_policy_num(self):
        new_active_policy_num = self._next_active_policy
        self.active_policy_indexes.append(new_active_policy_num)
        self._next_active_policy += 1
        assert len(set(self.active_policy_indexes)) == len(self.active_policy_indexes)
        assert len(set(self.fixed_policy_indexes)) == len(self.fixed_policy_indexes)
        return new_active_policy_num

    def move_active_policy_to_fixed(self, policy_num):
        if policy_num not in self.active_policy_indexes:
            raise ValueError(f"Player {self.player} Policy {policy_num} isn't an active policy. "
                             f"(Active policies are {self.active_policy_indexes})")
        if policy_num in self.fixed_policy_indexes:
            raise ValueError(f"Player {self.player} Policy {policy_num} is already fixed. "
                             f"(Fixed policies are {self.fixed_policy_indexes}, Active policies: {self.active_policy_indexes})")
        self.active_policy_indexes.remove(policy_num)
        self.fixed_policy_indexes.append(policy_num)
        self.fixed_policy_indexes.sort()
        assert len(set(self.active_policy_indexes)) == len(self.active_policy_indexes)
        assert len(set(self.fixed_policy_indexes)) == len(self.fixed_policy_indexes)
        all_indexes = list(set(self.active_policy_indexes).union(set(self.fixed_policy_indexes)))
        all_indexes.sort()
        assert np.array_equal(self.fixed_policy_indexes, list(range(0, policy_num + 1))), \
            f"all_indexes: {all_indexes}\n" \
            f"list(range(0, policy_num+1)):{list(range(0, policy_num + 1))}\n" \
            f"self.active_policy_indexes:{self.active_policy_indexes}\n" \
            f"self.fixed_policy_indexes:{self.fixed_policy_indexes}"


class P2SROManager(object):

    def __init__(self,
                 n_players,
                 is_two_player_symmetric_zero_sum: bool,
                 do_external_payoff_evals_for_new_fixed_policies: bool,
                 games_per_external_payoff_eval: int,
                 eval_dispatcher_port: int = 4536,
                 payoff_table_exponential_average_coeff: float = None,
                 get_manager_logger=None,
                 log_dir: str = None,
                 manager_metadata: dict = None):

        self._n_players = n_players
        self._is_two_player_symmetric_zero_sum = is_two_player_symmetric_zero_sum
        self._do_external_payoff_evals_for_new_fixed_policies = do_external_payoff_evals_for_new_fixed_policies
        self._games_per_external_payoff_eval = games_per_external_payoff_eval
        self._payoff_table = PayoffTable(n_players=n_players,
                                         exponential_average_coeff=payoff_table_exponential_average_coeff)

        num_player_stats = n_players
        self._player_stats = [_P2SROPlayerStats(player=i) for i in range(num_player_stats)]
        if self._is_two_player_symmetric_zero_sum:
            # both array entries point to the same stats
            self._player_stats[1] = self._player_stats[0]

        self._eval_dispatcher = EvalDispatcherWithServer(games_per_eval=self._games_per_external_payoff_eval,
                                                         game_is_two_player_symmetric=self._is_two_player_symmetric_zero_sum,
                                                         drop_duplicate_requests=True,
                                                         port=eval_dispatcher_port)
        self._eval_dispatcher.add_on_eval_result_callback(
            on_eval_result=lambda eval_result: self._on_finished_eval_result(eval_result))

        self._pending_spec_matchups_for_new_fixed_policies = {}

        if log_dir is None:
            log_dir = f"/tmp/p2sro_{datetime_str()}"
        self.log_dir = log_dir
        print(f"Manager log dir is {self.log_dir}")

        if get_manager_logger is None:
            self._manager_logger = SimpleP2SROManagerLogger(p2sro_manger=self, log_dir=self.log_dir)
        else:
            self._manager_logger = get_manager_logger(self)

        if manager_metadata is None:
            manager_metadata = {}
        manager_metadata["log_dir"] = self.get_log_dir()
        manager_metadata["n_players"] = self.n_players()
        is_jsonable, json_err = check_if_jsonable(check_dict=manager_metadata)
        if not is_jsonable:
            raise ValueError(f"manager_metadata must be JSON serializable. "
                             f"The following error occurred when trying to serialize it:\n{json_err}")
        self.manager_metadata = manager_metadata

        self._modification_lock = RLock()

    def n_players(self) -> int:
        return self._n_players

    def get_log_dir(self) -> str:
        return self.log_dir

    def get_manager_metadata(self) -> dict:
        return self.manager_metadata

    def claim_new_active_policy_for_player(self, player, new_policy_metadata_dict) -> StrategySpec:
        with self._modification_lock:
            if player < 0 or player >= self._n_players:
                raise ValueError(f"player {player} is out of range. Must be in [0, n_players).")
            if self._is_two_player_symmetric_zero_sum and player != 0:
                raise ValueError("Always use player 0 for two player symmetric game (plays both sides).")

            new_active_policy_num = self._player_stats[player].get_new_active_policy_num()
            new_strat_id = self._strat_id(player=player, policy_num=new_active_policy_num)
            new_policy_spec = self._payoff_table.add_new_pure_strategy(player=player,
                                                                       strategy_id=new_strat_id,
                                                                       metadata=new_policy_metadata_dict)
            if self._is_two_player_symmetric_zero_sum:
                # Add the same strategy for the second player to maintain a 2D payoff matrix
                new_policy_spec = self._payoff_table.add_new_pure_strategy(player=1,
                                                                           strategy_id=new_strat_id,
                                                                           metadata=new_policy_metadata_dict)
                assert np.array_equal(list(new_policy_spec.get_pure_strat_indexes().keys()), [0, 1])
                assert len(set(new_policy_spec.get_pure_strat_indexes().values())) == 1

            self._manager_logger.on_new_active_policy(player=player, new_policy_num=new_active_policy_num,
                                                      new_policy_spec=new_policy_spec)

            # print(f"active policy pure strat indexes are: {new_policy_spec._pure_strategy_indexes}")
            return new_policy_spec

    def submit_new_active_policy_metadata(self, player, policy_num, metadata_dict) -> StrategySpec:
        with self._modification_lock:
            if player < 0 or player >= self._n_players:
                raise ValueError(f"player {player} is out of range. Must be in [0, n_players).")
            if self._is_two_player_symmetric_zero_sum and player != 0:
                raise ValueError("Always use player 0 for two player symmetric game (plays both sides).")
            if policy_num not in self._player_stats[player].active_policy_indexes:
                raise ValueError(f"Policy {policy_num} isn't an active policy for player {player}.")

            active_policy_spec: StrategySpec = self._payoff_table.get_spec_for_strat_id(
                strat_id=self._strat_id(player=player, policy_num=policy_num))
            active_policy_spec.update_metadata(new_metadata=metadata_dict)

            self._manager_logger.on_new_active_policy_metadata(player=player, policy_num=policy_num,
                                                               new_policy_spec=active_policy_spec)

            return active_policy_spec

    def can_active_policy_be_set_as_fixed_now(self, player, policy_num) -> bool:
        with self._modification_lock:
            if player < 0 or player >= self._n_players:
                raise ValueError(f"player {player} is out of range. Must be in [0, n_players).")
            if self._is_two_player_symmetric_zero_sum and player != 0:
                raise ValueError("Always use player 0 for two player symmetric game (plays both sides).")
            if policy_num not in self._player_stats[player].active_policy_indexes:
                raise ValueError(f"Policy {policy_num} isn't active for player {player}.")

            for other_player, player_stats in enumerate(self._player_stats):
                if other_player == player and not self._is_two_player_symmetric_zero_sum:
                    continue
                for other_policy_num in range(0, policy_num):
                    if other_policy_num not in player_stats.fixed_policy_indexes:
                        return False
            return True

    def set_active_policy_as_fixed(self, player, policy_num, final_metadata_dict) -> StrategySpec:
        with self._modification_lock:
            if player < 0 or player >= self._n_players:
                raise ValueError(f"player {player} is out of range. Must be in [0, n_players).")
            if self._is_two_player_symmetric_zero_sum and player != 0:
                raise ValueError("Always use player 0 for two player symmetric game (plays both sides).")
            if policy_num not in self._player_stats[player].active_policy_indexes:
                raise ValueError(f"Policy {policy_num} isn't an active policy for player {player}.")
            if not self.can_active_policy_be_set_as_fixed_now(player=player, policy_num=policy_num):
                raise ValueError(f"Policy {policy_num} can't be set to fixed yet. Lower policies are still active.")

            fixed_policy_spec = self.submit_new_active_policy_metadata(player=player, policy_num=policy_num,
                                                                       metadata_dict=final_metadata_dict)

            fixed_policy_spec_matchups = []
            if self._do_external_payoff_evals_for_new_fixed_policies:
                # We'll move the policy to fixed later once we get back the eval payoff results for all matchups
                # against other fixed policies.
                fixed_policy_spec_matchups = self._get_all_opponent_policy_matchups(as_player=player,
                                                                                    as_policy_num=policy_num,
                                                                                    fixed_policies_only=True,
                                                                                    include_pending_active_policies=True)

                # print(f"policy matchups needed: {fixed_policy_spec_matchups}")

            if len(fixed_policy_spec_matchups) == 0:
                # Don't do additional evaluations for the policy, and set it as fixed now.
                self._player_stats[player].move_active_policy_to_fixed(policy_num=policy_num)

                self._manager_logger.on_active_policy_moved_to_fixed(player=player, policy_num=policy_num,
                                                                     fixed_policy_spec=fixed_policy_spec)
            else:
                # Track which eval matchups we will be waiting on.
                self._pending_spec_matchups_for_new_fixed_policies[(player, policy_num)] = set(
                    fixed_policy_spec_matchups)

                # Place external eval requests for all opponent matchups for this policy.
                for matchup in fixed_policy_spec_matchups:
                    self._eval_dispatcher.submit_eval_request(policy_specs_for_each_player=matchup)

            return fixed_policy_spec

    def get_copy_of_latest_data(self) -> (PayoffTable, List[List[int]], List[List[int]]):
        with self._modification_lock:
            active_policies_per_player = [stats.active_policy_indexes.copy() for stats in self._player_stats]
            fixed_policies_per_player = [stats.fixed_policy_indexes.copy() for stats in self._player_stats]
            return self._payoff_table.copy(), active_policies_per_player, fixed_policies_per_player

    def submit_empirical_payoff_result(self,
                                       policy_specs_for_each_player: Tuple[StrategySpec],
                                       payoffs_for_each_player: Tuple[float],
                                       games_played: int,
                                       override_all_previous_results: bool):
        with self._modification_lock:
            if len(policy_specs_for_each_player) != self._n_players:
                raise ValueError(f"policy_specs_for_each_player should be length {self._n_players} but it was length "
                                 f"{len(policy_specs_for_each_player)}.")
            if len(payoffs_for_each_player) != self._n_players:
                raise ValueError(f"payoffs_for_each_player should be length {self._n_players} but it was length "
                                 f"{len(payoffs_for_each_player)}.")
            if self._is_two_player_symmetric_zero_sum and payoffs_for_each_player[0] != -payoffs_for_each_player[1]:
                raise ValueError(f"Since the game is two-player symmetric, player 0's payoff should be the negative of "
                                 f"player 1's, however, payoffs submitted were {payoffs_for_each_player[0]} and "
                                 f"{payoffs_for_each_player[1]}.")

            strat_ids_for_each_player = [
                self._strat_id(player=player, policy_num=spec.pure_strat_index_for_player(player))
                for player, spec in enumerate(policy_specs_for_each_player)]

            for player in range(self._n_players):
                player_payoff_index = 0 if self._is_two_player_symmetric_zero_sum else player
                self._payoff_table.add_empirical_payoff_result(
                    as_player=player,
                    strat_ids_for_each_player=strat_ids_for_each_player,
                    payoff=payoffs_for_each_player[player_payoff_index],
                    games_played=games_played,
                    override_all_previous_results=override_all_previous_results)

                if self._is_two_player_symmetric_zero_sum:
                    self._payoff_table.add_empirical_payoff_result(
                        as_player=player,
                        strat_ids_for_each_player=strat_ids_for_each_player[::-1],
                        payoff=-payoffs_for_each_player[player_payoff_index],
                        games_played=games_played,
                        override_all_previous_results=override_all_previous_results)

            self._manager_logger.on_payoff_result(
                policy_specs_for_each_player=policy_specs_for_each_player,
                payoffs_for_each_player=payoffs_for_each_player,
                games_played=games_played,
                overrode_all_previous_results=override_all_previous_results
            )

    def request_external_eval(self, policy_specs_for_each_player):
        with self._modification_lock:
            if len(policy_specs_for_each_player) != self._n_players:
                raise ValueError(f"policy_specs_for_each_player should be length {self._n_players} but it was length "
                                 f"{len(policy_specs_for_each_player)}.")
            self._eval_dispatcher.submit_eval_request(policy_specs_for_each_player=policy_specs_for_each_player)

    def is_policy_fixed(self, player, policy_num):
        with self._modification_lock:
            if player < 0 or player >= self._n_players:
                raise ValueError(f"player {player} is out of range. Must be in [0, n_players).")
            if policy_num in self._player_stats[player].fixed_policy_indexes:
                return True
            elif policy_num in self._player_stats[player].active_policy_indexes:
                return False
            else:
                raise ValueError(f"Policy {policy_num} isn't a fixed or active policy for player {player}. "
                                 f"Fixed policies are {self._player_stats[player].fixed_policy_indexes} "
                                 f"and active policies are {self._player_stats[player].active_policy_indexes}.")

    def _strat_id(self, player, policy_num):
        if self._is_two_player_symmetric_zero_sum:
            player = 0
        return f"player_{player}_policy_{policy_num}"

    def _get_all_opponent_policy_matchups(self, as_player, as_policy_num, fixed_policies_only=False,
                                          include_pending_active_policies=True):
        spec_lists_per_player = []

        for spec_list_player in range(self._n_players):
            if spec_list_player == as_player:
                as_policy_spec = self._payoff_table.get_spec_for_player_and_pure_strat_index(
                    player=as_player, pure_strat_index=as_policy_num)
                spec_lists_per_player.append([as_policy_spec])
            else:
                policy_specs_for_other_player = []
                if fixed_policies_only:
                    policy_nums_to_consider = self._player_stats[spec_list_player].fixed_policy_indexes.copy()
                else:
                    policy_nums_to_consider = self._player_stats[spec_list_player].active_policy_indexes.copy() + \
                                              self._player_stats[spec_list_player].fixed_policy_indexes.copy()

                if include_pending_active_policies:
                    for pending_player, pending_policy_num in self._pending_spec_matchups_for_new_fixed_policies.keys():
                        if pending_player == spec_list_player and pending_policy_num not in policy_nums_to_consider:
                            policy_nums_to_consider.append(pending_policy_num)

                if self._is_two_player_symmetric_zero_sum and as_policy_num in policy_nums_to_consider:
                    policy_nums_to_consider.remove(as_policy_num)

                for other_policy_num in policy_nums_to_consider:
                    other_policy_spec = self._payoff_table.get_spec_for_player_and_pure_strat_index(
                        player=spec_list_player, pure_strat_index=other_policy_num)
                    policy_specs_for_other_player.append(other_policy_spec)
                spec_lists_per_player.append(policy_specs_for_other_player)

        all_opponent_matchups = list(product(*spec_lists_per_player))
        return all_opponent_matchups

    def _on_finished_eval_result(self, eval_result: EvalResult):
        with self._modification_lock:
            eval_result_should_override_previous_results = False

            # print(f"current pending spec matchups: {self._pending_spec_matchups_for_new_fixed_policies}")

            # Check if we're waiting on this eval matchup to get the final payoff results for an active policy that
            # we're waiting to move to fixed.
            keys_to_remove = []
            for key, matchups_for_fixed_policy in self._pending_spec_matchups_for_new_fixed_policies.items():
                player, new_fixed_policy_num = key
                if eval_result.policy_specs_for_each_player in matchups_for_fixed_policy:
                    # We were waiting on this matchup, so remove it from the set of matchups we're waiting on.
                    matchups_for_fixed_policy.remove(eval_result.policy_specs_for_each_player)

                    # This is a final payoff eval for a fixed policy against fixed policy opponent(s).
                    # Override any other data currently on the payoff table for the same matchup.
                    eval_result_should_override_previous_results = True

                    if len(matchups_for_fixed_policy) == 0:
                        # We're no longer waiting on any eval results to move this policy to fixed.
                        keys_to_remove.append(key)

                        # Move it to fixed.
                        self._player_stats[player].move_active_policy_to_fixed(policy_num=new_fixed_policy_num)
                        fixed_policy_spec = self._payoff_table.get_spec_for_player_and_pure_strat_index(
                            player=player, pure_strat_index=new_fixed_policy_num)
                        self._manager_logger.on_active_policy_moved_to_fixed(player=player,
                                                                             policy_num=new_fixed_policy_num,
                                                                             fixed_policy_spec=fixed_policy_spec)
            for key in keys_to_remove:
                del self._pending_spec_matchups_for_new_fixed_policies[key]
            # print(f"new pending spec matchups: {self._pending_spec_matchups_for_new_fixed_policies}")

            # Add this payoff result to our payoff table.
            self.submit_empirical_payoff_result(
                policy_specs_for_each_player=eval_result.policy_specs_for_each_player,
                payoffs_for_each_player=eval_result.payoff_for_each_player,
                games_played=eval_result.games_played,
                override_all_previous_results=eval_result_should_override_previous_results
            )
