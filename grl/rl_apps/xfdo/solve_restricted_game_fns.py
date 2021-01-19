from abc import ABC, abstractmethod

import numpy as np
from copy import deepcopy
from threading import RLock
from itertools import product
from typing import List, Tuple, Union, Dict, Callable
import json
import os
from typing import Callable
from grl.p2sro.payoff_table import PayoffTable, PayoffTableStrategySpec
from grl.utils import datetime_str
from grl.rl_apps.scenarios.stopping_conditions import StoppingCondition, TwoPlayerBRRewardsBelowAmtStoppingCondition

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


class SolveRestrictedGameDynamicRewardThreshold(SolveRestrictedGame):

    def __init__(self, scenario: dict,
                 get_reward_threshold: Callable[[Dict[int, List[PayoffTableStrategySpec]]], float],
                 min_episodes: int, required_fields: List[str]):
        self.scenario = scenario
        self.get_reward_threshold = get_reward_threshold
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
            stop_if_br_avg_rew_falls_below=self.get_reward_threshold(br_spec_lists_for_each_player),
            min_episodes=self.min_episodes,
            required_fields_in_last_train_iter=self.required_fields
        )

        return _solve_game(scenario=self.scenario, log_dir=log_dir,
                           br_spec_lists_for_each_player=br_spec_lists_for_each_player,
                           stopping_condition=stopping_condition)