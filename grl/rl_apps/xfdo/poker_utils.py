from typing import Callable, List, Tuple, Union

import numpy as np
import pyspiel
from open_spiel.python.algorithms.exploitability import exploitability
from open_spiel.python.policy import tabular_policy_from_callable
from pyspiel import Game as OpenSpielGame
from ray.rllib.policy import Policy

from grl.envs.poker_multi_agent_env import parse_discrete_poker_action_from_continuous_space
from grl.rl_apps.psro.poker_utils import tabular_policies_from_weighted_policies, JointPlayerPolicy, softmax
from grl.algos.xfdo.action_space_conversion import RestrictedToBaseGameActionSpaceConverter
from grl.algos.xfdo.opnsl_restricted_game import AgentRestrictedGameOpenSpielObsConversions


def _parse_action_probs_from_action_info(action, action_info, legal_actions_list, total_num_discrete_actions):
    action_probs = None
    for key in ['policy_targets', 'action_probs']:
        if key in action_info:
            action_probs = action_info[key]
            break
    if action_probs is None:
        if "behaviour_logits" in action_info:
            action_logits = action_info['behaviour_logits']
            action_probs = softmax(action_logits)
        else:
            # assume action is continuous and is to be quntized to nearest legal action
            discrete_action = parse_discrete_poker_action_from_continuous_space(continuous_action=action,
                                                                                legal_actions_list=legal_actions_list,
                                                                                total_num_discrete_actions_including_dummy=total_num_discrete_actions)
            action_probs = np.zeros(shape=total_num_discrete_actions, dtype=np.float32)
            action_probs[discrete_action] = 1.0

    return action_probs


def openspiel_policy_from_nonlstm_rllib_xfdo_policy(openspiel_game: OpenSpielGame,
                                                    rllib_policy: Policy,
                                                    restricted_game_convertor: Union[
                                                        RestrictedToBaseGameActionSpaceConverter, AgentRestrictedGameOpenSpielObsConversions]):
    is_openspiel_restricted_game = isinstance(restricted_game_convertor, AgentRestrictedGameOpenSpielObsConversions)

    def policy_callable(state: pyspiel.State):

        valid_actions_mask = state.legal_actions_mask()
        legal_actions_list = state.legal_actions()

        # assert np.array_equal(valid_actions, np.ones_like(valid_actions)) # should be always true at least for Kuhn

        info_state_vector = state.information_state_tensor()

        if openspiel_game.get_type().short_name in ["leduc_poker", "oshi_zumo", "oshi_zumo_tiny",
                                                    "universal_poker"] or is_openspiel_restricted_game:
            # Observation includes both the info_state and legal actions, but agent isn't forced to take legal actions.
            # Taking an illegal action will result in a random legal action being played.
            # Allows easy compatibility with standard RL implementations for small action-space games like this one.
            obs = np.concatenate(
                (np.asarray(info_state_vector, dtype=np.float32), np.asarray(valid_actions_mask, dtype=np.float32)),
                axis=0)
        else:
            obs = np.asarray(info_state_vector, dtype=np.float32)

        if is_openspiel_restricted_game:
            os_restricted_game_convertor: AgentRestrictedGameOpenSpielObsConversions = restricted_game_convertor
            try:
                obs = os_restricted_game_convertor.orig_obs_to_restricted_game_obs[tuple(obs)]
            except KeyError:
                print(
                    f"missing key: {tuple(obs)}\nexample key: {list(os_restricted_game_convertor.orig_obs_to_restricted_game_obs.keys())[0]}")
                raise
        action, _, restricted_action_info = rllib_policy.compute_single_action(obs=obs, state=[], explore=False)
        restricted_game_action_probs = _parse_action_probs_from_action_info(action=action,
                                                                            action_info=restricted_action_info,
                                                                            legal_actions_list=legal_actions_list,
                                                                            total_num_discrete_actions=len(
                                                                                valid_actions_mask))

        if is_openspiel_restricted_game:
            action_probs = restricted_game_action_probs
        else:
            base_action_probs_for_each_restricted_game_action = []
            for restricted_game_action in range(len(restricted_game_action_probs)):
                _, _, action_info = restricted_game_convertor.get_base_game_action(
                    obs=obs, restricted_game_action=restricted_game_action, use_delegate_policy_exploration=False,
                    clip_base_game_actions=False, delegate_policy_state=None
                )
                base_game_action_probs_for_rstr_action = _parse_action_probs_from_action_info(action=action,
                                                                                              action_info=action_info,
                                                                                              legal_actions_list=legal_actions_list,
                                                                                              total_num_discrete_actions=len(
                                                                                                  valid_actions_mask))
                base_action_probs_for_each_restricted_game_action.append(base_game_action_probs_for_rstr_action)

            action_probs = np.zeros_like(base_action_probs_for_each_restricted_game_action[0])
            for base_action_probs, restricted_action_prob in zip(base_action_probs_for_each_restricted_game_action,
                                                                 restricted_game_action_probs):
                action_probs += (base_action_probs * restricted_action_prob)

        assert np.isclose(sum(action_probs), 1.0)

        if len(action_probs) > len(valid_actions_mask) and len(action_probs) % len(valid_actions_mask) == 0:
            # we may be using a dummy action variant of poker
            dummy_action_probs = action_probs.copy()
            action_probs = np.zeros_like(valid_actions_mask, dtype=np.float64)
            for i, action_prob in enumerate(dummy_action_probs):
                action_probs[i % len(valid_actions_mask)] += action_prob
            assert np.isclose(sum(action_probs), 1.0)

        # Since the rl env will execute a random legal action if an illegal action is chosen, redistribute probability
        # of choosing an illegal action evenly across all legal actions.
        # num_legal_actions = sum(valid_actions_mask)
        # if num_legal_actions > 0:
        #     total_legal_action_probability = sum(action_probs * valid_actions_mask)
        #     total_illegal_action_probability = 1.0 - total_legal_action_probability
        #     action_probs = (action_probs + (total_illegal_action_probability / num_legal_actions)) * valid_actions_mask

        assert np.isclose(sum(action_probs), 1.0)

        legal_action_probs = []
        valid_action_prob_sum = 0.0
        for idx in range(len(valid_actions_mask)):
            if valid_actions_mask[idx] == 1.0:
                legal_action_probs.append(action_probs[idx])
                valid_action_prob_sum += action_probs[idx]
        assert np.isclose(valid_action_prob_sum, 1.0)

        return {action_name: action_prob for action_name, action_prob in zip(legal_actions_list, legal_action_probs)}

    # callable_policy = PolicyFromCallable(game=openspiel_game, callable_policy=policy_callable)

    # convert to tabular policy in case the rllib policy changes after this function is called
    return tabular_policy_from_callable(game=openspiel_game, callable_policy=policy_callable)


def xfdo_nfsp_measure_exploitability_nonlstm(rllib_policies: List[Policy],
                                             restricted_game_convertors: Union[
                                                 List[RestrictedToBaseGameActionSpaceConverter], List[
                                                     AgentRestrictedGameOpenSpielObsConversions]],
                                             poker_game_version: str,
                                             open_spiel_env_config: dict = None):
    if open_spiel_env_config is None:
        if poker_game_version in ["kuhn_poker", "leduc_poker"]:
            open_spiel_env_config = {
                "players": pyspiel.GameParameter(2)
            }
        elif poker_game_version in ["oshi_zumo_tiny"]:
            poker_game_version = "oshi_zumo"
            open_spiel_env_config = {
                "coins": pyspiel.GameParameter(6),
                "size": pyspiel.GameParameter(2),
                "horizon": pyspiel.GameParameter(8),
            }
        else:
            open_spiel_env_config = {}

    open_spiel_env_config = {k: pyspiel.GameParameter(v) if not isinstance(v, pyspiel.GameParameter) else v for k, v in
                             open_spiel_env_config.items()}

    openspiel_game = pyspiel.load_game(poker_game_version, open_spiel_env_config)

    opnsl_policies = []
    assert isinstance(restricted_game_convertors, list)
    for action_space_converter, rllib_policy in zip(restricted_game_convertors, rllib_policies):
        openspiel_policy = openspiel_policy_from_nonlstm_rllib_xfdo_policy(openspiel_game=openspiel_game,
                                                                           rllib_policy=rllib_policy,
                                                                           restricted_game_convertor=action_space_converter
                                                                           )
        opnsl_policies.append(openspiel_policy)

    nfsp_policy = JointPlayerPolicy(game=openspiel_game, policies=opnsl_policies)

    # Exploitability is NashConv / num_players
    if poker_game_version == "universal_poker":
        print("Measuring exploitability for universal_poker policy. This will take a while...")
    exploitability_result = exploitability(game=openspiel_game, policy=nfsp_policy)
    return exploitability_result


def xfdo_snfsp_measure_exploitability_nonlstm(br_checkpoint_path_tuple_list: List[Tuple[str, str]],
                                              set_policy_weights_fn: Callable,
                                              rllib_policies: List[Policy],
                                              restricted_game_convertors: Union[
                                                  List[RestrictedToBaseGameActionSpaceConverter], List[
                                                      AgentRestrictedGameOpenSpielObsConversions]],
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
            assert isinstance(restricted_game_convertors, list)
            for player, (restricted_game_convertor, player_rllib_policy) in enumerate(
                    zip(restricted_game_convertors, rllib_policies)):
                checkpoint_path = checkpoint_path_tuple[player]
                set_policy_weights_fn(player_rllib_policy, checkpoint_path)

                single_openspiel_policy = openspiel_policy_from_nonlstm_rllib_xfdo_policy(openspiel_game=openspiel_game,
                                                                                          rllib_policy=player_rllib_policy,
                                                                                          restricted_game_convertor=restricted_game_convertor)
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
    if poker_game_version == "universal_poker":
        print("Measuring exploitability for universal_poker policy. This will take a while...")
    exploitability_result = exploitability(game=openspiel_game, policy=nfsp_policy)
    return exploitability_result
