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


class SolveRestrictedGameStaticStoppingCondition(SolveRestrictedGame):

    def __init__(self,
                 fn_to_actually_sovle_game,
                 get_stopping_condition: Callable[[], StoppingCondition]):
        raise NotImplementedError()

    def __call__(self, log_dir: str,
                 br_spec_lists_for_each_player: Dict[int, List[PayoffTableStrategySpec]]) -> RestrictedGameSolveResult:
        raise NotImplementedError()


class SolveRestrictedGameLoweringThresholdWithBRApproxExploitability(SolveRestrictedGame):

    def __init__(self,
                 fn_to_actually_sovle_game,
                 get_threshold_given_exploitability: Callable[[float], float],
                 get_stopping_condition_given_threshold: Callable[[float], StoppingCondition]):
        raise NotImplementedError()

    def __call__(self, log_dir: str,
                 br_spec_lists_for_each_player: Dict[int, List[PayoffTableStrategySpec]]) -> RestrictedGameSolveResult:
        raise NotImplementedError()