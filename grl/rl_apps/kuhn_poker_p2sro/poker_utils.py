from grl.p2sro.payoff_table import PayoffTable, PayoffTableStrategySpec
from grl.rllib_tools.leduc_dqn.valid_actions_fcnet import LeducDQNFullyConnectedNetwork
from ray.rllib.agents.sac import SACTorchPolicy
from ray.rllib.agents.trainer import with_common_config, MODEL_DEFAULTS
from ray.rllib.policy import Policy
from ray.rllib.models.action_dist import ActionDistribution

import deepdish

from typing import Iterable, Dict, Callable, List, Tuple, Generator
from open_spiel.python import rl_environment

def _calculate_metanash_and_exploitability_of_fixed_policies(payoff_table: PayoffTable, player_policy_nums: dict):
    pass


import numpy as np


import ray

from open_spiel.python.policy import Policy as OpenSpielPolicy, PolicyFromCallable, TabularPolicy, tabular_policy_from_policy
from open_spiel.python.algorithms.exploitability import nash_conv, exploitability
from pyspiel import Game as OpenSpielGame

import pyspiel
from grl.p2sro.p2sro_manager.utils import get_latest_metanash_strategies
from grl.p2sro.payoff_table import PayoffTableStrategySpec
from grl.rl_apps.kuhn_poker_p2sro.poker_multi_agent_env import PokerMultiAgentEnv
from ray.rllib.utils import merge_dicts, try_import_torch
from ray.rllib.utils.typing import ModelGradients, ModelWeights, \
    TensorType, TrainerConfigDict, AgentID, PolicyID
torch, _ = try_import_torch()

def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def _callable_tabular_policy(tabular_policy):
    """Turns a tabular policy into a callable.

    Args:
    tabular_policy: A dictionary mapping information state key to a dictionary
      of action probabilities (action -> prob).

    Returns:
    A function `state` -> list of (action, prob)
    """

    def wrap(state):
        infostate_key = state.information_state(state.current_player())
        assert infostate_key in tabular_policy, f"tabular_policy: {tabular_policy}, key: {infostate_key}"
        ap_list = []
        for action in state.legal_actions():
            assert action in tabular_policy[infostate_key]
            ap_list.append((action, tabular_policy[infostate_key][action]))

        return ap_list

    return wrap


def _policy_dict_at_state(callable_policy, state):
    """Turns a policy function into a dictionary at a specific state.

    Args:
    callable_policy: A function from `state` -> lis of (action, prob),
    state: the specific state to extract the policy from.

    Returns:
    A dictionary of action -> prob at this state.
    """

    infostate_policy_list = callable_policy(state)
    if isinstance(infostate_policy_list, dict):
        infostate_policy = infostate_policy_list
        legal_actions = state.legal_actions()
        for action in infostate_policy.keys():
            assert action in legal_actions
        for action in legal_actions:
            if action not in infostate_policy:
                infostate_policy[action] = 0.0
    else:
        infostate_policy = {}
        for ap in infostate_policy_list:
            infostate_policy[ap[0]] = ap[1]
    return infostate_policy


def _recursively_update_average_policies(state, avg_reach_probs, br_reach_probs,
                                         avg_policies,
                                         best_responses,
                                         alpha,
                                         avg_policy_tables,
                                         delta_tolerance=1e-10):
    """Recursive implementation of the average strategy update."""

    if state.is_terminal():
        return
    elif state.is_chance_node():
        for action, _ in state.chance_outcomes():
            new_state = state.clone()
            new_state.apply_action(action)
            _recursively_update_average_policies(state=new_state,
                                                 avg_reach_probs=avg_reach_probs,
                                                 br_reach_probs=br_reach_probs,
                                                 avg_policies=avg_policies,
                                                 best_responses=best_responses,
                                                 alpha=alpha,
                                                 avg_policy_tables=avg_policy_tables,
                                                 delta_tolerance=delta_tolerance)
    else:
        player = state.current_player()

        # print(f"avg policy: {avg_policies[player]}, state: {state}")

        avg_policy = _policy_dict_at_state(avg_policies[player], state)
        br_policy = _policy_dict_at_state(best_responses[player], state)
        legal_actions = state.legal_actions()
        infostate_key = state.information_state(player)
        # First traverse the subtrees.
        for action in legal_actions:
            assert action in br_policy, f"action is {action}, br policy is {br_policy}"
            assert action in avg_policy
            new_state = state.clone()
            new_state.apply_action(action)
            new_avg_reach = np.copy(avg_reach_probs)
            new_avg_reach[player] *= avg_policy[action]
            new_br_reach = np.copy(br_reach_probs)
            new_br_reach[player] *= br_policy[action]
            _recursively_update_average_policies(state=new_state,
                                                                 avg_reach_probs=new_avg_reach,
                                                                 br_reach_probs=new_br_reach,
                                                                 avg_policies=avg_policies,
                                                                 best_responses=best_responses,
                                                                 alpha=alpha,
                                                                 avg_policy_tables=avg_policy_tables,
                                                                 delta_tolerance=delta_tolerance)
        # Now, do the updates.
        if infostate_key not in avg_policy_tables[player]:
            # alpha = 1 / (self._iterations + 1)
            avg_policy_tables[player][infostate_key] = {}
            pr_sum = 0.0

            if avg_reach_probs[player] + br_reach_probs[player] > 0.0:
                for action in legal_actions:

                    assert isinstance(br_policy[action], float), f"br policy action : {br_policy[action]}"
                    assert isinstance(avg_policy[action], float), f"avg policy action : {avg_policy[action]}"
                    assert isinstance(alpha[player], float), f"alpha[player]: {alpha[player]}"
                    assert isinstance(br_reach_probs[player], float), f"br_reach_probs[player]: {br_reach_probs[player]}"
                    # assert br_reach_probs[player] != 0, f"br_reach_probs[player]: {br_reach_probs[player]}"

                    pr = (
                      avg_policy[action] + (alpha[player] * br_reach_probs[player] *
                                          (br_policy[action] - avg_policy[action])) /
                      ((1.0 - alpha[player]) * avg_reach_probs[player] +
                       alpha[player] * br_reach_probs[player]))

                    assert not np.isnan(pr)
                    assert isinstance(pr, float), f"pr is {pr}"

                    avg_policy_tables[player][infostate_key][action] = pr
                    pr_sum += pr
                # print(f"pr_sum: {pr_sum}")
                assert (1.0 - delta_tolerance <= pr_sum <= 1.0 + delta_tolerance)
            else:
                for action in legal_actions:
                    avg_policy_tables[player][infostate_key][action] = 1.0/len(legal_actions)

def tabular_policies_from_weighted_policies(game: OpenSpielGame,
                                          policy_iterable,
                                          weights: List[Tuple[float, float]]):
    """Converts multiple Policy instances into an weighted averaged TabularPolicy.

    Args:
      game: The game for which we want a TabularPolicy.
      policy_iterable: for each player, an iterable that returns tuples of Openspiel policies
      weights: for each player, probabilities of selecting actions from each policy

    Returns:
      A averaged OpenSpiel Policy over the policy_iterable.
    """
    num_players = game.num_players()
    # A set of callables that take in a state and return a list of
    # (action, probability) tuples.
    avg_policies = [None, None]
    total_weights_added = np.zeros(num_players)
    for index, (best_responses, weights_for_each_br) in enumerate(zip(policy_iterable, weights)):
        weights_for_each_br = np.asarray(weights_for_each_br, dtype=np.float64)
        # print(f"best responses: {best_responses}")
        # print(f"total_weights_added: {total_weights_added}, weights_for_each_br: {weights_for_each_br}")
        total_weights_added += weights_for_each_br
        if index == 0:
            for i in range(num_players):
                avg_policies[i] = tabular_policy_from_policy(game=game, policy=best_responses[i])
        else:
            br_reach_probs = np.ones(num_players)
            avg_reach_probs = np.ones(num_players)
            average_policy_tables = [{} for _ in range(num_players)]
            _recursively_update_average_policies(state=game.new_initial_state(),
                                                 avg_reach_probs=avg_reach_probs,
                                                 br_reach_probs=br_reach_probs,
                                                 avg_policies=avg_policies,
                                                 best_responses=best_responses,
                                                 alpha=weights_for_each_br/total_weights_added,
                                                 avg_policy_tables=average_policy_tables)
            for i in range(num_players):
                avg_policies[i] = _callable_tabular_policy(average_policy_tables[i])

    for i in range(num_players):
        avg_policies[i] = PolicyFromCallable(game=game, callable_policy=avg_policies[i])

    # print(f"avg_policies: {avg_policies}")
    return avg_policies

def openspiel_policy_from_nonlstm_rllib_policy(openspiel_game: OpenSpielGame,
                                               rllib_policy: Policy):

    def policy_callable(state: pyspiel.State):

        valid_actions_mask = state.legal_actions_mask()
        legal_actions_list = state.legal_actions()

        # assert np.array_equal(valid_actions, np.ones_like(valid_actions)) # should be always true at least for Kuhn

        info_state_vector = state.information_state_as_normalized_vector()

        if openspiel_game.get_type().short_name == "leduc_poker":
            # Observation includes both the info_state and legal actions, but agent isn't forced to take legal actions.
            # Taking an illegal action will result in a random legal action being played.
            # Allows easy compatibility with standard RL implementations for small action-space games like this one.
            obs = np.concatenate(
                (np.asarray(info_state_vector, dtype=np.float32), np.asarray(valid_actions_mask, dtype=np.float32)),
                axis=0)
        else:
            obs = np.asarray(info_state_vector, dtype=np.float32)

        _, _, action_info = rllib_policy.compute_single_action(obs=obs, state=[], explore=False)

        action_probs = None
        for key in ['policy_targets', 'action_probs']:
            if key in action_info:
                action_probs = action_info[key]
                break
        if action_probs is None:
            action_logits = action_info['behaviour_logits']
            action_probs = softmax(action_logits)

        if len(action_probs) > len(valid_actions_mask) and len(action_probs) % len(valid_actions_mask) == 0:
            # we may be using a dummy action variant of poker
            dummy_action_probs = action_probs.copy()
            action_probs = np.zeros_like(valid_actions_mask)
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
        assert np.isclose(valid_action_prob_sum, 1.0), (action_probs, valid_actions_mask, action_info.get('behaviour_logits'))

        return {action_name: action_prob for action_name, action_prob in zip(legal_actions_list, legal_action_probs)}

    callable_policy = PolicyFromCallable(game=openspiel_game, callable_policy=policy_callable)

    # convert to tabular policy in case the rllib policy changes after this function is called
    return tabular_policy_from_policy(game=openspiel_game, policy=callable_policy)

def measure_exploitability_nonlstm(rllib_policy: Policy,
                                   poker_game_version: str,
                                   policy_mixture_dict: Dict[PayoffTableStrategySpec, float] = None,
                                   set_policy_weights_fn: Callable[[PayoffTableStrategySpec], None] = None):
    if poker_game_version in ["kuhn_poker", "leduc_poker"]:
        open_spiel_env_config = {
            "players": pyspiel.GameParameter(2)
        }
    else:
        open_spiel_env_config = {}

    openspiel_game = pyspiel.load_game(poker_game_version, open_spiel_env_config)

    if policy_mixture_dict is None:
        openspiel_policy = openspiel_policy_from_nonlstm_rllib_policy(openspiel_game=openspiel_game,
                                                                      rllib_policy=rllib_policy)
    else:
        if set_policy_weights_fn is None:
            raise ValueError("If policy_mixture_dict is passed a value, a set_policy_weights_fn must be passed as well.")

        def policy_iterable():
            for policy_spec in policy_mixture_dict.keys():
                set_policy_weights_fn(policy_spec)

                single_openspiel_policy = openspiel_policy_from_nonlstm_rllib_policy(openspiel_game=openspiel_game,
                                                                              rllib_policy=rllib_policy)
                yield single_openspiel_policy

        openspiel_policy = tabular_policy_from_weighted_policies(game=openspiel_game,
                                                                 policy_iterable=policy_iterable(),
                                                                 weights=policy_mixture_dict.values())
    # Exploitability is NashConv / num_players
    exploitability_result = exploitability(game=openspiel_game, policy=openspiel_policy)
    return exploitability_result


class JointPlayerPolicy(OpenSpielPolicy):
  """Joint policy to be evaluated."""

  def __init__(self, game, policies):
    player_ids = [0, 1]
    super(JointPlayerPolicy, self).__init__(game, player_ids)
    self._policies = policies
    self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (
        state.information_state_as_normalized_vector(cur_player))
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None)

    player_policy: OpenSpielPolicy = self._policies[cur_player]

    return player_policy.action_probabilities(state=state, player_id=cur_player)


def psro_measure_exploitability_nonlstm(br_checkpoint_path_tuple_list: List[Tuple[str, str]],
                                        metanash_weights: List[Tuple[float, float]],
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

    # print(f"weights: {metanash_weights}")

    avg_policies = tabular_policies_from_weighted_policies(game=openspiel_game,
                                                           policy_iterable=policy_iterable(),
                                                           weights=metanash_weights)

    nfsp_policy = JointPlayerPolicy(game=openspiel_game, policies=avg_policies)

    # Exploitability is NashConv / num_players
    exploitability_result = exploitability(game=openspiel_game, policy=nfsp_policy)
    return exploitability_result



def get_stats_for_single_payoff_table(payoff_table:PayoffTable, highest_policy_num: int, poker_env_config, policy_class, policy_config):

    ray.init(ignore_reinit_error=True, local_mode=True, num_cpus=1)

    poker_game_version = poker_env_config["version"]
    temp_env = PokerMultiAgentEnv(env_config=poker_env_config)

    # def fetch_logits(policy):
    #     return {
    #         "behaviour_logits": policy.model.last_output(),
    #     }
    #
    # policy_class = policy_class.with_updates(
    #     extra_action_fetches_fn=fetch_logits
    # )

    def extra_action_out_fn(policy: Policy, input_dict, state_batches, model,
                            action_dist: ActionDistribution) -> Dict[str, TensorType]:
        action = action_dist.deterministic_sample()
        action_probs = torch.zeros_like(policy.q_values).long()
        action_probs[0][action[0]] = 1.0
        return {"q_values": policy.q_values, "action_probs": action_probs}
    policy_class = policy_class.with_updates(
        extra_action_out_fn=extra_action_out_fn
    )

    policies = [policy_class(
        obs_space=temp_env.observation_space,
        action_space=temp_env.action_space,
        config=policy_config) for _ in range(2)]
        # config=with_common_config({
        #     'model': model_config,
        #     'env': PokerMultiAgentEnv,
        #     'env_config': poker_env_config
        # }
        # )
    # ) for _ in range(2)]

    if poker_game_version == "leduc_poker":
        assert isinstance(policies[0].model, LeducDQNFullyConnectedNetwork)

    def set_policy_weights(policy: Policy, checkpoint_path: str):
        checkpoint_data = deepdish.io.load(path=checkpoint_path)
        weights = checkpoint_data["weights"]
        weights = {k.replace("_dot_", "."): v for k, v in weights.items()}
        policy.set_weights(weights)

    exploitability_per_generation = []
    total_steps_per_generation = []
    total_episodes_per_generation = []
    num_policies_per_generation = []

    for i, n_policies in enumerate(range(1, highest_policy_num + 1)):

        metanash_probs_0 = get_latest_metanash_strategies(payoff_table=payoff_table,
                                                        as_player=1,
                                                        as_policy_num=n_policies,
                                                        fictitious_play_iters=2000,
                                                        mix_with_uniform_dist_coeff=0.0)[0].probabilities_for_each_strategy()

        metanash_probs_1 = get_latest_metanash_strategies(payoff_table=payoff_table,
                                                          as_player=0,
                                                          as_policy_num=n_policies,
                                                          fictitious_play_iters=2000,
                                                          mix_with_uniform_dist_coeff=0.0)[1].probabilities_for_each_strategy()

        pure_strat_index = get_latest_metanash_strategies(payoff_table=payoff_table,
                                       as_player=0,
                                       as_policy_num=n_policies,
                                       fictitious_play_iters=2000,
                                       mix_with_uniform_dist_coeff=0.0)[1].sample_policy_spec().get_pure_strat_indexes()
        # print(f"pure strat index: {pure_strat_index}")


        policy_specs_0 = payoff_table.get_ordered_spec_list_for_player(player=0)[:n_policies]

        policy_specs_1 = payoff_table.get_ordered_spec_list_for_player(player=1)[:n_policies]

        assert len(metanash_probs_1) == len(policy_specs_1), f"len(metanash_probs_1): {len(metanash_probs_1)}, len(policy_specs_1): {len(policy_specs_1)}"
        assert len(metanash_probs_0) == len(policy_specs_0)
        assert len(policy_specs_0) == len(policy_specs_1)

        br_checkpoint_paths = []
        metanash_weights = []

        # print(policy_specs_0)
        # print(metanash_probs_0)
        # print(policy_specs_1)
        # print(metanash_probs_1)

        for spec_0, prob_0, spec_1, prob_1 in zip(policy_specs_0, metanash_probs_0, policy_specs_1, metanash_probs_1):
            br_checkpoint_paths.append((spec_0.metadata["checkpoint_path"], spec_1.metadata["checkpoint_path"]))
            metanash_weights.append((prob_0, prob_1))

        exploitability_this_gen = psro_measure_exploitability_nonlstm(
            br_checkpoint_path_tuple_list=br_checkpoint_paths,
            metanash_weights=metanash_weights,
            set_policy_weights_fn=set_policy_weights,
            rllib_policies=policies,
            poker_game_version=poker_game_version
        )

        print(f"{n_policies} policies, {exploitability_this_gen} exploitability")

        policy_spec_added_this_gen = [payoff_table.get_spec_for_player_and_pure_strat_index(
            player=p, pure_strat_index=i) for p in range(2)]

        latest_policy_steps = sum(policy_spec_added_this_gen[p].metadata["timesteps_training_br"] for p in range(2))
        latest_policy_episodes = sum(policy_spec_added_this_gen[p].metadata["episodes_training_br"] for p in range(2))

        if i > 0:
            total_steps_this_generation = latest_policy_steps + total_steps_per_generation[i-1]
            total_episodes_this_generation = latest_policy_episodes + total_episodes_per_generation[i-1]
        else:
            total_steps_this_generation = latest_policy_steps
            total_episodes_this_generation = latest_policy_episodes

        exploitability_per_generation.append(exploitability_this_gen)
        total_steps_per_generation.append(total_steps_this_generation)
        total_episodes_per_generation.append(total_episodes_this_generation)
        num_policies_per_generation.append(n_policies)

    stats_out = {'num_policies': num_policies_per_generation, 'exploitability': exploitability_per_generation,
                 'timesteps': total_steps_per_generation, 'episodes': total_episodes_per_generation}

    return stats_out
