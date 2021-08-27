from typing import List

import pyspiel
from open_spiel.python.algorithms.exploitability import exploitability
from ray.rllib.policy import Policy

from grl.rl_apps.psro.poker_utils import openspiel_policy_from_nonlstm_rllib_policy, JointPlayerPolicy


def nfsp_measure_exploitability_nonlstm(rllib_policies: List[Policy],
                                        poker_game_version: str,
                                        open_spiel_env_config: dict = None):
    if open_spiel_env_config is None:
        if poker_game_version in ["kuhn_poker", "leduc_poker"]:
            open_spiel_env_config = {
                "players": pyspiel.GameParameter(2)
            }
        else:
            open_spiel_env_config = {}

    open_spiel_env_config = {k: pyspiel.GameParameter(v) if not isinstance(v, pyspiel.GameParameter) else v for k, v in
                             open_spiel_env_config.items()}

    openspiel_game = pyspiel.load_game(poker_game_version, open_spiel_env_config)
    if poker_game_version == "oshi_zumo":
        openspiel_game = pyspiel.convert_to_turn_based(openspiel_game)

    opnsl_policies = []
    for rllib_policy in rllib_policies:
        openspiel_policy = openspiel_policy_from_nonlstm_rllib_policy(openspiel_game=openspiel_game,
                                                                      rllib_policy=rllib_policy,
                                                                      game_version=poker_game_version,
                                                                      game_parameters=open_spiel_env_config,
        )
        opnsl_policies.append(openspiel_policy)

    nfsp_policy = JointPlayerPolicy(game=openspiel_game, policies=opnsl_policies)

    # Exploitability is NashConv / num_players
    if poker_game_version == "universal_poker":
        print("Measuring exploitability for universal_poker policy. This will take a while...")
    exploitability_result = exploitability(game=openspiel_game, policy=nfsp_policy)
    return exploitability_result
