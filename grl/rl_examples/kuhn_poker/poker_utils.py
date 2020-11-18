from grl.payoff_table import PayoffTable, PayoffTableStrategySpec
from ray.rllib.agents.sac import SACTorchPolicy
from ray.rllib.agents.trainer import with_common_config, MODEL_DEFAULTS

import deepdish

def _calculate_metanash_and_exploitability_of_fixed_policies(payoff_table: PayoffTable, player_policy_nums: dict):
    pass


import numpy as np


import ray

from open_spiel.python.policy import Policy as OSPolicy, PolicyFromCallable, TabularPolicy
from open_spiel.python.algorithms.exploitability import nash_conv, exploitability

import pyspiel
from grl.p2sro_manager.utils import get_latest_metanash_strategies

from grl.rl_examples.kuhn_poker.poker_multi_agent_env import PokerMultiAgentEnv

def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def tabular_policy_from_weighted_policies(game, policy_iterable, weights):
    """Converts multiple Policy instances into an weighted averaged TabularPolicy.

    Args:
      game: The game for which we want a TabularPolicy.
      policy_iterable: An iterable that returns Openspiel policies
      weights: probabilities of selecting actions from each policy

    Returns:
      A averaged TabularPolicy over the policy_iterable.
    """

    empty_tabular_policy = TabularPolicy(game)

    assert np.isclose(1.0, sum(weights))

    # initially zero out all policy probs
    for state_index, state in enumerate(empty_tabular_policy.states):
        infostate_policy = [
            0
            for _ in range(game.num_distinct_actions())
        ]
        empty_tabular_policy.action_probability_array[
        state_index, :] = infostate_policy

    # add weighted probs from each policy we're averaging over
    for policy, weight, in zip(policy_iterable, weights):
        for state_index, state in enumerate(empty_tabular_policy.states):
            old_action_probabilities = empty_tabular_policy.action_probabilities(state)
            add_action_probabilities = policy.action_probabilities(state)
            infostate_policy = [
                old_action_probabilities.get(action, 0.) + add_action_probabilities.get(action, 0.) * weight
                for action in range(game.num_distinct_actions())
            ]
            empty_tabular_policy.action_probability_array[
            state_index, :] = infostate_policy

    # check that all action probs pers state add up to one in the newly created policy
    for state_index, state in enumerate(empty_tabular_policy.states):
        action_probabilities = empty_tabular_policy.action_probabilities(state)
        infostate_policy = [
            action_probabilities.get(action, 0.)
            for action in range(game.num_distinct_actions())
        ]

        assert np.isclose(1.0, sum(infostate_policy)), "INFOSTATE POLICY: {}".format(infostate_policy)

    return empty_tabular_policy


def openspiel_policy_from_nonlstm_rllib_policy(openspiel_game, poker_game_version, rllib_policy):

    def policy_callable(state: pyspiel.State):

        valid_actions_mask = state.legal_actions_mask()
        legal_actions_list = state.legal_actions()

        # assert np.array_equal(valid_actions, np.ones_like(valid_actions)) # should be always true at least for Kuhn

        info_state_vector = state.information_state_as_normalized_vector()

        # Observation includes both the info_state and legal actions, but agent isn't forced to take legal actions.
        # Taking an illegal action will result in a random legal action being played.
        # Allows easy compatibility with standard RL implementations for small action-space games like this one.
        obs = np.concatenate(
            (np.asarray(info_state_vector, dtype=np.float32), np.asarray(valid_actions_mask, dtype=np.float32)),
            axis=0)

        _, _, action_info = rllib_policy.compute_single_action(obs=obs, state=[])

        action_probs = None
        for key in ['policy_targets', 'action_probs']:
            if key in action_info:
                action_probs = action_info[key]
                break
        if action_probs is None:
            action_logits = action_info['behaviour_logits']
            action_probs = softmax(action_logits)

        # Since the rl env will execute a random legal action if an illegal action is chosen, redistribute probability
        # of choosing an illegal action evenly across all legal actions.
        num_legal_actions = sum(valid_actions_mask)
        if num_legal_actions > 0:
            total_legal_action_probability = sum(action_probs * valid_actions_mask)
            total_illegal_action_probability = 1.0 - total_legal_action_probability
            action_probs = (action_probs + (total_illegal_action_probability / num_legal_actions)) * valid_actions_mask
            assert np.isclose(sum(action_probs), 1.0)

        legal_action_probs = []
        for idx in range(len(valid_actions_mask)):
            if valid_actions_mask[idx] == 1.0:
                legal_action_probs.append(action_probs[idx])

        return {action_name: action_prob for action_name, action_prob in zip(legal_actions_list, legal_action_probs)}

    return PolicyFromCallable(game=openspiel_game, callable_policy=policy_callable)


def measure_exploitability_nonlstm(rllib_policy, poker_game_version, policy_mixture_dict=None, set_policy_weights_fn=None):
    if poker_game_version in ["kuhn_poker", "leduc_poker"]:
        open_spiel_env_config = {
            "players": pyspiel.GameParameter(2)
        }
    else:
        open_spiel_env_config = {}

    openspiel_game = pyspiel.load_game(poker_game_version, open_spiel_env_config)

    if policy_mixture_dict is None:
        openspiel_policy = openspiel_policy_from_nonlstm_rllib_policy(openspiel_game=openspiel_game,
                                                                      poker_game_version=poker_game_version,
                                                                      rllib_policy=rllib_policy)
    else:
        if set_policy_weights_fn is None:
            raise ValueError("If policy_mixture_dict is passed a value, a set_policy_weights_fn must be passed as well.")

        def policy_iterable():
            for policy_spec in policy_mixture_dict.keys():
                set_policy_weights_fn(policy_spec)

                single_openspiel_policy = openspiel_policy_from_nonlstm_rllib_policy(openspiel_game=openspiel_game,
                                                                              poker_game_version=poker_game_version,
                                                                              rllib_policy=rllib_policy)
                yield single_openspiel_policy

        openspiel_policy = tabular_policy_from_weighted_policies(game=openspiel_game,
                                                                 policy_iterable=policy_iterable(),
                                                                 weights=policy_mixture_dict.values())
    # Exploitability is NashConv / num_players
    exploitability_result = exploitability(game=openspiel_game, policy=openspiel_policy)
    return exploitability_result



def get_stats_for_single_payoff_table(payoff_table:PayoffTable, highest_policy_num: int, poker_game_version, policy_class, model_config):
    poker_env_config = {
        'version': poker_game_version,
    }

    ray.init(ignore_reinit_error=True, local_mode=True)

    temp_env = PokerMultiAgentEnv(env_config=poker_env_config)

    def fetch_logits(policy):
        return {
            "behaviour_logits": policy.model.last_output(),
        }

    policy_class = policy_class.with_updates(
        extra_action_fetches_fn=fetch_logits
    )

    policy = policy_class(
        obs_space=temp_env.observation_space,
        action_space=temp_env.action_space,
        config=with_common_config({
            'model': model_config,
            'env': PokerMultiAgentEnv,
            'env_config': poker_env_config
        })
    )

    def set_policy_weights(policy_spec: PayoffTableStrategySpec):
        checkpoint_path = policy_spec.metadata["checkpoint_path"]
        checkpoint_data = deepdish.io.load(path=checkpoint_path)
        policy.set_weights(checkpoint_data["weights"])

    exploitability_per_generation = []
    total_steps_per_generation = []
    total_episodes_per_generation = []
    num_policies_per_generation = []

    for i, n_policies in enumerate(range(1, highest_policy_num + 1)):

        metanash_probs = get_latest_metanash_strategies(payoff_table=payoff_table,
                                                        as_player=0,
                                                        as_policy_num=highest_policy_num + 1,
                                                        fictitious_play_iters=20000,
                                                        mix_with_uniform_dist_coeff=0.0)[0]

        policy_specs = payoff_table.get_ordered_spec_list_for_player(player=0)

        policy_dict = {spec: prob for spec, prob in zip(policy_specs, metanash_probs)}

        exploitability_this_gen = measure_exploitability_nonlstm(rllib_policy=policy,
                                  poker_game_version=poker_game_version,
                                  policy_mixture_dict=policy_dict,
                                  set_policy_weights_fn=set_policy_weights)

        print(f"{n_policies} policies, {exploitability_this_gen} exploitability")

        policy_spec_added_this_gen = payoff_table.get_spec_for_player_and_pure_strat_index(player=0, pure_strat_index=i)

        latest_policy_steps = policy_spec_added_this_gen.metadata["timesteps_training_br"]
        latest_policy_episodes = policy_spec_added_this_gen.metadata["episodes_training_br"]

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
                 'total_steps': total_steps_per_generation, 'total_episodes': total_episodes_per_generation}

    return stats_out
