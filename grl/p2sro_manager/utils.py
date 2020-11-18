import numpy as np
from typing import Dict, List, Tuple
from grl.p2sro_manager.p2sro_manager import P2SROManager
from grl.payoff_table import PayoffTable, PayoffTableStrategySpec


class PolicySpecDistribution(object):

    def __init__(self, payoff_table: PayoffTable, player: int, policy_selection_probs_indexed_by_policy_num: List[float]):
        self._probs_to_policy_specs = {
            selection_prob: payoff_table.get_spec_for_player_and_pure_strat_index(
                player=player, pure_strat_index=policy_num)
            for policy_num, selection_prob in enumerate(policy_selection_probs_indexed_by_policy_num)
        }

    def sample_policy_spec(self) -> PayoffTableStrategySpec:
        return np.random.choice(a=list(self._probs_to_policy_specs.values()),
                                p=list(self._probs_to_policy_specs.keys()))

    def probabilities_for_each_strategy(self) -> np.ndarray:
        return np.asarray(list(self._probs_to_policy_specs.keys()), dtype=np.float64)


def get_latest_metanash_strategies(payoff_table: PayoffTable,
                                   as_player: int,
                                   as_policy_num: int,
                                   fictitious_play_iters: int,
                                   mix_with_uniform_dist_coeff: float = 0.0) -> Dict[int, PolicySpecDistribution]:

    # Currently this function only handles 2-player games

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
        assert len(player_payoff_matrix.shape) == 2 # assume a 2D payoff matrix

        # only consider policies below 'as_policy_num' in the p2sro hierarchy
        player_payoff_matrix = player_payoff_matrix[:as_policy_num, :as_policy_num]

        player_payoff_matrix_current_player_is_rows = player_payoff_matrix.transpose((as_player, other_player))
        averages, exps = _fictitious_play(iters=fictitious_play_iters,
                                          payoffs=player_payoff_matrix_current_player_is_rows)
        selection_probs = np.copy(averages[-1])

        if mix_with_uniform_dist_coeff is not None and mix_with_uniform_dist_coeff > 0:
            uniform_dist = np.ones_like(selection_probs) / len(selection_probs)
            selection_probs = mix_with_uniform_dist_coeff * uniform_dist + (
                    1.0 - mix_with_uniform_dist_coeff) * selection_probs

        opponent_strategy_distributions[other_player] = PolicySpecDistribution(
            payoff_table=payoff_table, player=other_player,
            policy_selection_probs_indexed_by_policy_num=selection_probs)

    return opponent_strategy_distributions


def _get_br_to_strat(strat, payoffs):
    row_weighted_payouts = strat @ payoffs
    br = np.zeros_like(row_weighted_payouts)
    br[np.argmin(row_weighted_payouts)] = 1
    return br


def _fictitious_play(iters, payoffs):
    dim = len(payoffs)
    pop = np.random.uniform(0, 1, (1, dim))
    pop = pop / pop.sum(axis=1)[:, None]
    averages = pop
    exps = []
    for i in range(iters):
        average = np.average(pop, axis=0)
        br = _get_br_to_strat(average, payoffs=payoffs)
        exp1 = average @ payoffs @ br.T
        exp2 = br @ payoffs @ average.T
        exps.append(exp2 - exp1)
        averages = np.vstack((averages, average))
        pop = np.vstack((pop, br))
    return averages, exps