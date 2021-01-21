import logging
import argparse
import os
import grl
from grl.p2sro.p2sro_manager import P2SROManagerWithServer, SimpleP2SROManagerLogger
from grl.utils import datetime_str, ensure_dir
from grl.rl_apps.scenarios.poker import scenarios
from grl.rl_apps.kuhn_poker_p2sro.general_psro_eval import launch_evals

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str)
    commandline_args = parser.parse_args()

    scenario_name = commandline_args.scenario
    try:
        scenario = scenarios[scenario_name]
    except KeyError:
        raise NotImplementedError(f"Unknown scenario name: \'{scenario_name}\'. Existing scenarios are:\n"
                                  f"{list(scenarios.keys())}")

    default_psro_port = scenario["psro_port"]
    default_eval_port = scenario["eval_port"]
    games_per_payoff_eval = scenario["games_per_payoff_eval"]
    p2sro = scenario["p2sro"]

    if p2sro:
        raise NotImplementedError("a little more setup is needed in configs to launch p2sro this way")

    log_dir = os.path.join(os.path.dirname(grl.__file__), "data", scenario_name, f"manager_{datetime_str()}")

    name_file_path = os.path.join(log_dir, "scenario_name.txt")
    ensure_dir(name_file_path)
    with open(name_file_path, "w+") as name_file:
        name_file.write(scenario_name)

    manager = P2SROManagerWithServer(
        port=os.getenv("P2SRO_PORT", default_psro_port),
        eval_dispatcher_port=os.getenv("EVAL_PORT", default_eval_port),
        n_players=2,
        is_two_player_symmetric_zero_sum=False,
        do_external_payoff_evals_for_new_fixed_policies=True,
        games_per_external_payoff_eval=games_per_payoff_eval,
        payoff_table_exponential_average_coeff=None,
        log_dir=log_dir
    )
    print(f"Launched P2SRO Manager with server.")

    launch_evals(scenario_name=scenario_name, block=False)

    print(f"Launched evals")

    manager.wait_for_server_termination()
