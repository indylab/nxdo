from typing import Dict, List

import numpy as np

from grl.algos.p2sro.payoff_table import PayoffTable
from grl.utils.strategy_spec import StrategySpec


class PolicySpecDistribution(object):

    def __init__(self, payoff_table: PayoffTable, player: int,
                 policy_selection_probs_indexed_by_policy_num: List[float]):
        self._probs_to_policy_specs = {
            selection_prob: payoff_table.get_spec_for_player_and_pure_strat_index(
                player=player, pure_strat_index=policy_num)
            for policy_num, selection_prob in enumerate(policy_selection_probs_indexed_by_policy_num)
        }
        self.player = player

    def sample_policy_spec(self) -> StrategySpec:
        return np.random.choice(a=list(self._probs_to_policy_specs.values()),
                                p=list(self._probs_to_policy_specs.keys()))

    def probabilities_for_each_strategy(self) -> np.ndarray:
        return np.asarray(list(self._probs_to_policy_specs.keys()), dtype=np.float64)


def get_latest_metanash_strategies(payoff_table: PayoffTable,
                                   as_player: int,
                                   as_policy_num: int,
                                   fictitious_play_iters: int,
                                   mix_with_uniform_dist_coeff: float = 0.0,
                                   print_matrix: bool = True) -> Dict[int, PolicySpecDistribution]:
    # Currently this function only handles 2-player games
    if as_policy_num is None:
        as_policy_num = payoff_table.shape()[as_player] - 1

    if not 0 <= as_player < payoff_table.n_players():
        raise ValueError(f"as_player {as_player} should be in the range [0, {payoff_table.n_players()}).")
    if payoff_table.shape()[as_player] <= as_policy_num:
        raise ValueError(f"In the payoff table, policy_num {as_policy_num} is out of range for player {as_player}.")

    if payoff_table.n_players() != 2:
        raise NotImplemented("Solving normal form Nash equilibrium strats for >2 player games not implemented.")

    if as_policy_num == 0:
        return None

    other_players = list(range(0, payoff_table.n_players()))
    other_players.remove(as_player)

    opponent_strategy_distributions = {}
    for other_player in other_players:
        player_payoff_matrix = payoff_table.get_payoff_matrix_for_player(player=other_player)
        assert len(player_payoff_matrix.shape) == 2  # assume a 2D payoff matrix

        # only consider policies below 'as_policy_num' in the p2sro hierarchy
        player_payoff_matrix = player_payoff_matrix[:as_policy_num, :as_policy_num]

        player_payoff_matrix_current_player_is_rows = player_payoff_matrix.transpose((other_player, as_player))

        if print_matrix:
            print(f"payoff matrix as {other_player} (row) against {as_player} (columns):")
            print(player_payoff_matrix_current_player_is_rows)

        row_averages, col_averages, exps = fictitious_play(iters=fictitious_play_iters,
                                                           payoffs=player_payoff_matrix_current_player_is_rows)
        selection_probs = np.copy(row_averages[-1])

        if mix_with_uniform_dist_coeff is not None and mix_with_uniform_dist_coeff > 0:
            uniform_dist = np.ones_like(selection_probs) / len(selection_probs)
            selection_probs = mix_with_uniform_dist_coeff * uniform_dist + (
                    1.0 - mix_with_uniform_dist_coeff) * selection_probs

        opponent_strategy_distributions[other_player] = PolicySpecDistribution(
            payoff_table=payoff_table, player=other_player,
            policy_selection_probs_indexed_by_policy_num=selection_probs)

    return opponent_strategy_distributions


def get_br_to_strat(strat, payoffs, strat_is_row=True, verbose=False):
    if strat_is_row:
        weighted_payouts = strat @ payoffs
        if verbose:
            print("strat ", strat)
            print("weighted payouts ", weighted_payouts)

        br = np.zeros_like(weighted_payouts)
        br[np.argmin(weighted_payouts)] = 1
        idx = np.argmin(weighted_payouts)
    else:
        weighted_payouts = payoffs @ strat.T
        if verbose:
            print("strat ", strat)
            print("weighted payouts ", weighted_payouts)

        br = np.zeros_like(weighted_payouts)
        br[np.argmax(weighted_payouts)] = 1
        idx = np.argmax(weighted_payouts)
    return br, idx


def fictitious_play(payoffs, iters=2000, verbose=False):
    row_dim = payoffs.shape[0]
    col_dim = payoffs.shape[1]
    row_pop = np.random.uniform(0, 1, (1, row_dim))
    row_pop = row_pop / row_pop.sum(axis=1)[:, None]
    row_averages = row_pop
    col_pop = np.random.uniform(0, 1, (1, col_dim))
    col_pop = col_pop / col_pop.sum(axis=1)[:, None]
    col_averages = col_pop
    exps = []
    for i in range(iters):
        row_average = np.average(row_pop, axis=0)
        col_average = np.average(col_pop, axis=0)

        row_br, idx = get_br_to_strat(col_average, payoffs, strat_is_row=False, verbose=False)
        col_br, idx = get_br_to_strat(row_average, payoffs, strat_is_row=True, verbose=False)

        exp1 = row_average @ payoffs @ col_br.T
        exp2 = row_br @ payoffs @ col_average.T
        exps.append(exp2 - exp1)
        if verbose:
            print(exps[-1], "exploitability")

        row_averages = np.vstack((row_averages, row_average))
        col_averages = np.vstack((col_averages, col_average))

        row_pop = np.vstack((row_pop, row_br))
        col_pop = np.vstack((col_pop, col_br))
    return row_averages, col_averages, exps
