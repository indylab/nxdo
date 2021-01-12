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

from open_spiel.python.policy import Policy as OpenSpielPolicy, PolicyFromCallable, TabularPolicy, tabular_policy_from_policy
from open_spiel.python.algorithms.exploitability import nash_conv, exploitability
from pyspiel import Game as OpenSpielGame

import pyspiel
from grl.p2sro.p2sro_manager.utils import get_latest_metanash_strategies
from grl.p2sro.payoff_table import PayoffTableStrategySpec
from grl.rl_apps.kuhn_poker_p2sro.poker_multi_agent_env import PokerMultiAgentEnv
from grl.rl_apps.kuhn_poker_p2sro.poker_utils import tabular_policies_from_weighted_policies, JointPlayerPolicy, softmax
from grl.xfdo.action_space_conversion import RestrictedToBaseGameActionSpaceConverter


def _parse_action_probs_from_action_info(action_info):
    action_probs = None
    for key in ['policy_targets', 'action_probs']:
        if key in action_info:
            action_probs = action_info[key]
            break
    if action_probs is None:
        action_logits = action_info['behaviour_logits']
        action_probs = softmax(action_logits)
    return action_probs


def openspiel_policy_from_nonlstm_rllib_xfdo_policy(openspiel_game: OpenSpielGame,
                                               rllib_policy: Policy,
                                                    action_space_converter: RestrictedToBaseGameActionSpaceConverter):

    def policy_callable(state: pyspiel.State):

        valid_actions_mask = state.legal_actions_mask()
        legal_actions_list = state.legal_actions()

        # assert np.array_equal(valid_actions, np.ones_like(valid_actions)) # should be always true at least for Kuhn

        info_state_vector = state.information_state_as_normalized_vector()

        # Observation includes both the info_state and legal actions, but agent isn't forced to take legal actions.
        # Taking an illegal action will result in a random legal action being played.
        # Allows easy compatibility with standard RL implementations for small action-space games like this one.
        # obs = np.concatenate(
        #     (np.asarray(info_state_vector, dtype=np.float32), np.asarray(valid_actions_mask, dtype=np.float32)),
        #     axis=0)
        obs = np.asarray(info_state_vector, dtype=np.float32)

        _, _, restricted_action_info = rllib_policy.compute_single_action(obs=obs, state=[], explore=False)
        restricted_game_action_probs = _parse_action_probs_from_action_info(action_info=restricted_action_info)

        base_action_probs_for_each_restricted_game_action = []
        for restricted_game_action in range(len(restricted_game_action_probs)):
            _, _, action_info = action_space_converter.get_base_game_action(
                obs=obs, restricted_game_action=restricted_game_action, use_delegate_policy_exploration=False,
                clip_base_game_actions=False, delegate_policy_state=None
            )
            base_game_action_probs_for_rstr_action = _parse_action_probs_from_action_info(action_info=action_info)
            base_action_probs_for_each_restricted_game_action.append(base_game_action_probs_for_rstr_action)

        action_probs = np.zeros(shape=len(valid_actions_mask))
        for base_action_probs, restricted_action_prob in zip(base_action_probs_for_each_restricted_game_action, restricted_game_action_probs):
            action_probs += (base_action_probs * restricted_action_prob)
        assert np.isclose(sum(action_probs), 1.0)

        # Since the rl env will execute a random legal action if an illegal action is chosen, redistribute probability
        # of choosing an illegal action evenly across all legal actions.
        num_legal_actions = sum(valid_actions_mask)
        if num_legal_actions > 0:
            total_legal_action_probability = sum(action_probs * valid_actions_mask)
            total_illegal_action_probability = 1.0 - total_legal_action_probability
            action_probs = (action_probs + (total_illegal_action_probability / num_legal_actions)) * valid_actions_mask

        assert np.isclose(sum(action_probs), 1.0)

        legal_action_probs = []
        valid_action_prob_sum = 0.0
        for idx in range(len(valid_actions_mask)):
            if valid_actions_mask[idx] == 1.0:
                legal_action_probs.append(action_probs[idx])
                valid_action_prob_sum += action_probs[idx]
        assert np.isclose(valid_action_prob_sum, 1.0)

        return {action_name: action_prob for action_name, action_prob in zip(legal_actions_list, legal_action_probs)}

    callable_policy = PolicyFromCallable(game=openspiel_game, callable_policy=policy_callable)

    # convert to tabular policy in case the rllib policy changes after this function is called
    return tabular_policy_from_policy(game=openspiel_game, policy=callable_policy)



def xfdo_nfsp_measure_exploitability_nonlstm(rllib_policies: List[Policy],
                                             action_space_converters: List[RestrictedToBaseGameActionSpaceConverter],
                                   poker_game_version: str):
    if poker_game_version in ["kuhn_poker", "leduc_poker"]:
        open_spiel_env_config = {
            "players": pyspiel.GameParameter(2)
        }
    else:
        open_spiel_env_config = {}

    openspiel_game = pyspiel.load_game(poker_game_version, open_spiel_env_config)

    opnsl_policies = []
    for action_space_converter, rllib_policy in zip(action_space_converters, rllib_policies):
        openspiel_policy = openspiel_policy_from_nonlstm_rllib_xfdo_policy(openspiel_game=openspiel_game,
                                                                          rllib_policy=rllib_policy,
                                                                           action_space_converter=action_space_converter
                                                                      )
        opnsl_policies.append(openspiel_policy)

    nfsp_policy = JointPlayerPolicy(game=openspiel_game, policies=opnsl_policies)

    # Exploitability is NashConv / num_players
    exploitability_result = exploitability(game=openspiel_game, policy=nfsp_policy)
    return exploitability_result



def xfdo_snfsp_measure_exploitability_nonlstm(br_checkpoint_path_tuple_list: List[Tuple[str, str]],
                                         set_policy_weights_fn: Callable,
                                        rllib_policies: List[Policy],
                                        action_space_converters: List[RestrictedToBaseGameActionSpaceConverter],
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
            for player, (action_space_converter, player_rllib_policy) in enumerate(zip(action_space_converters, rllib_policies)):
                checkpoint_path = checkpoint_path_tuple[player]
                set_policy_weights_fn(player_rllib_policy, checkpoint_path)

                single_openspiel_policy = openspiel_policy_from_nonlstm_rllib_xfdo_policy(openspiel_game=openspiel_game,
                                                                                     rllib_policy=player_rllib_policy,
                                                                                        action_space_converter=action_space_converter)
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