import numpy as np

from typing import List, Dict

from grl.p2sro.payoff_table import PayoffTableStrategySpec
from grl.rl_apps.scenarios.stopping_conditions import TwoPlayerBRRewardsBelowAmtStoppingCondition

from grl.xfdo.xfdo_manager.manager import SolveRestrictedGame, RestrictedGameSolveResult

from grl.rl_apps.xfdo.general_xfdo_nfsp_metanash import train_off_policy_rl_nfsp_restricted_game
from grl.rl_apps.xfdo.general_xfdo_cfp_metanash import train_cfp_restricted_game


def _solve_game(scenario: dict,
                log_dir: str,
                br_spec_lists_for_each_player: Dict[int, List[PayoffTableStrategySpec]],
                stopping_condition) -> RestrictedGameSolveResult:

    if scenario["xfdo_metanash_method"] == "nfsp":
        avg_policy_specs, final_train_result = train_off_policy_rl_nfsp_restricted_game(
            results_dir=log_dir, scenario=scenario, player_to_base_game_action_specs=br_spec_lists_for_each_player,
            stopping_condition=stopping_condition, print_train_results=True
        )
    elif scenario["xfdo_metanash_method"] == "cfp":
        avg_policy_specs, final_train_result = train_cfp_restricted_game(
            results_dir=log_dir, scenario=scenario,
            player_to_base_game_action_specs=br_spec_lists_for_each_player,
            stopping_condition=stopping_condition, print_train_results=True
        )
    else:
        raise NotImplementedError(f"Unknown xfdo_metanash_method: {scenario['xfdo_metanash_method']}")

    if scenario["calculate_openspiel_metanash"]:
        extra_data = {"exploitability": final_train_result["z_avg_policy_exploitability"]}
    else:
        extra_data = {}

    return RestrictedGameSolveResult(latest_metanash_spec_for_each_player=tuple(avg_policy_specs),
                                     episodes_spent_in_solve=final_train_result["episodes_total"],
                                     timesteps_spent_in_solve=final_train_result["timesteps_total"],
                                     extra_data_to_log=extra_data)


class SolveRestrictedGameFixedRewardThreshold(SolveRestrictedGame):

    def __init__(self, scenario: dict, br_reward_threshold: float, min_episodes: int, required_fields: List[str]):

        self.scenario = scenario
        self.br_reward_threshold = br_reward_threshold
        self.min_episodes = min_episodes

        if required_fields is None:
            required_fields = []
        if scenario["calculate_openspiel_metanash"] and "z_avg_policy_exploitability" not in required_fields:
            required_fields.append("z_avg_policy_exploitability")

        self.required_fields = required_fields

    def __call__(self,
                 log_dir: str,
                 br_spec_lists_for_each_player: Dict[int, List[PayoffTableStrategySpec]]) -> RestrictedGameSolveResult:

        stopping_condition = TwoPlayerBRRewardsBelowAmtStoppingCondition(
            stop_if_br_avg_rew_falls_below=self.br_reward_threshold,
            min_episodes=self.min_episodes,
            required_fields_in_last_train_iter=self.required_fields
        )

        return _solve_game(scenario=self.scenario, log_dir=log_dir,
                           br_spec_lists_for_each_player=br_spec_lists_for_each_player,
                           stopping_condition=stopping_condition)


class SolveRestrictedGameDynamicRewardThreshold1(SolveRestrictedGame):

    def __init__(self,
                 scenario: dict,
                 dont_solve_first_n_xfdo_iters: int,
                 starting_rew_threshold: float,
                 min_rew_threshold: float,
                 min_episodes: int, required_fields: List[str]):

        self.scenario = scenario
        self._min_episodes = min_episodes
        self._current_rew_threshold = starting_rew_threshold
        self._min_rew_threshold = min_rew_threshold

        self._current_xfdo_iter = 0
        self._dont_solve_first_n_xfdo_iters = dont_solve_first_n_xfdo_iters

        if required_fields is None:
            required_fields = []
        if scenario["calculate_openspiel_metanash"] and "z_avg_policy_exploitability" not in required_fields:
            required_fields.append("z_avg_policy_exploitability")
        self.required_fields = required_fields

    def _update_reward_threshold(self, br_spec_lists_for_each_player: Dict[int, List[PayoffTableStrategySpec]]) -> float:
        latest_avg_br_reward = float(np.mean(
            [spec_list[-1].metadata["average_br_reward"] for spec_list in br_spec_lists_for_each_player.values()])
        )
        # Halve the current threshold if the latest br had a lower reward than it.
        # Current threshold cannot go below the min_threshold value.
        if latest_avg_br_reward < self._current_rew_threshold:
            self._current_rew_threshold = max(self._min_rew_threshold, self._current_rew_threshold / 2.0)

    def __call__(self,
                 log_dir: str,
                 br_spec_lists_for_each_player: Dict[int, List[PayoffTableStrategySpec]]) -> RestrictedGameSolveResult:
        # This method is called each time we need to solve the metanash for XFDO

        if self._current_xfdo_iter < self._dont_solve_first_n_xfdo_iters:
            # Instantly get back an untrained metanash (an untrained avg policy network in the case of NFSP)
            stopping_condition = TwoPlayerBRRewardsBelowAmtStoppingCondition(
                stop_if_br_avg_rew_falls_below=100000.0,
                min_episodes=0,
                required_fields_in_last_train_iter=self.required_fields
            )
        else:
            self._update_reward_threshold(br_spec_lists_for_each_player=br_spec_lists_for_each_player)

            stopping_condition = TwoPlayerBRRewardsBelowAmtStoppingCondition(
                stop_if_br_avg_rew_falls_below=self._current_rew_threshold,
                min_episodes=self._min_episodes,
                required_fields_in_last_train_iter=self.required_fields
            )

        self._current_xfdo_iter += 1

        return _solve_game(scenario=self.scenario,
                           log_dir=log_dir,
                           br_spec_lists_for_each_player=br_spec_lists_for_each_player,
                           stopping_condition=stopping_condition)
