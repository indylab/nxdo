from grl.p2sro.payoff_table import PayoffTable, PayoffTableStrategySpec
from ray.rllib.agents.sac import SACTorchPolicy
from ray.rllib.agents.trainer import with_common_config, MODEL_DEFAULTS
from ray.rllib.policy import Policy
import deepdish

from typing import Iterable, Dict, Callable, List, Tuple, Generator
from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
import numpy as np


import ray

from open_spiel.python.policy import Policy as OpenSpielPolicy, PolicyFromCallable, TabularPolicy
from open_spiel.python.algorithms.exploitability import nash_conv, exploitability
from pyspiel import Game as OpenSpielGame

import pyspiel
from grl.p2sro.p2sro_manager.utils import get_latest_metanash_strategies
from grl.p2sro.payoff_table import PayoffTableStrategySpec
from grl.rl_examples.kuhn_poker_p2sro.poker_multi_agent_env import PokerMultiAgentEnv
from grl.rl_examples.kuhn_poker_p2sro.poker_utils import openspiel_policy_from_nonlstm_rllib_policy, tabular_policies_from_weighted_policies, _policy_dict_at_state, JointPlayerPolicy





def nfsp_measure_exploitability_nonlstm(rllib_policies: List[Policy],
                                   poker_game_version: str):
    if poker_game_version in ["kuhn_poker", "leduc_poker"]:
        open_spiel_env_config = {
            "players": pyspiel.GameParameter(2)
        }
    else:
        open_spiel_env_config = {}

    openspiel_game = pyspiel.load_game(poker_game_version, open_spiel_env_config)

    opnsl_policies = []
    for rllib_policy in rllib_policies:
        openspiel_policy = openspiel_policy_from_nonlstm_rllib_policy(openspiel_game=openspiel_game,
                                                                          rllib_policy=rllib_policy)
        opnsl_policies.append(openspiel_policy)

    nfsp_policy = JointPlayerPolicy(game=openspiel_game, policies=opnsl_policies)

    # Exploitability is NashConv / num_players
    exploitability_result = exploitability(game=openspiel_game, policy=nfsp_policy)
    return exploitability_result



def snfsp_measure_exploitability_nonlstm(br_checkpoint_path_tuple_list: List[Tuple[str, str]],
                                         set_policy_weights_fn: Callable,
                                        rllib_policies: List[Policy],
                                        poker_game_version: str):
    if poker_game_version in ["kuhn_poker", "leduc_poker"]:
        open_spiel_env_config = {
            "players": pyspiel.GameParameter(2)
        }
    else:
        open_spiel_env_config = {}

    openspiel_game = pyspiel.load_game(poker_game_version, open_spiel_env_config)

    def policy_iterable():
        for checkpoint_path_tuple in br_checkpoint_path_tuple_list:
            openspiel_policies = []
            for player, player_rllib_policy in enumerate(rllib_policies):
                checkpoint_path = checkpoint_path_tuple[player]
                set_policy_weights_fn(player_rllib_policy, checkpoint_path)

                single_openspiel_policy = openspiel_policy_from_nonlstm_rllib_policy(openspiel_game=openspiel_game,
                                                                                     rllib_policy=player_rllib_policy)
                openspiel_policies.append(single_openspiel_policy)
            yield openspiel_policies

    num_players = 2
    weights = np.ones(shape=(len(br_checkpoint_path_tuple_list), num_players)) / len(br_checkpoint_path_tuple_list)

    print(f"weights: {weights}")

    avg_policies = tabular_policies_from_weighted_policies(game=openspiel_game,
                                                           policy_iterable=policy_iterable(),
                                                           weights=weights)

    print(f"avg_policies: {avg_policies}")

    nfsp_policy = JointPlayerPolicy(game=openspiel_game, policies=avg_policies)

    # Exploitability is NashConv / num_players
    exploitability_result = exploitability(game=openspiel_game, policy=nfsp_policy)
    return exploitability_result