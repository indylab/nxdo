import argparse
import logging
import os

import grl
from grl.p2sro.p2sro_manager import P2SROManagerWithServer
from grl.rl_apps.psro.general_psro_eval import launch_evals
from grl.rl_apps.scenarios import scenario_catalog, PSROScenario
from grl.rl_apps.scenarios.ray_setup import init_ray_for_scenario
from grl.rl_apps import GRL_SEED
from grl.utils.common import datetime_str, ensure_dir
from grl.utils.port_listings import establish_new_server_port_for_service

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str)
    parser.add_argument('--psro_port', type=int, required=False, default=None)
    parser.add_argument('--eval_port', type=int, required=False, default=None)
    commandline_args = parser.parse_args()

    scenario_name = commandline_args.scenario
    scenario: PSROScenario = scenario_catalog.get(scenario_name=scenario_name)

    psro_port = commandline_args.psro_port
    if psro_port is None:
        psro_port = establish_new_server_port_for_service(service_name=f"seed_{GRL_SEED}_{scenario.name}")

    eval_port = commandline_args.eval_port
    if eval_port is None:
        eval_port = establish_new_server_port_for_service(service_name=f"seed_{GRL_SEED}_{scenario.name}_evals")

    ray_head_address = init_ray_for_scenario(scenario=scenario, head_address=None, logging_level=logging.INFO)

    if scenario.p2sro:
        raise NotImplementedError("a little more setup is needed in configs to launch p2sro this way")

    log_dir = os.path.join(os.path.dirname(grl.__file__), "data", scenario_name, f"manager_{datetime_str()}")

    name_file_path = os.path.join(log_dir, "scenario_name.txt")
    ensure_dir(name_file_path)
    with open(name_file_path, "w+") as name_file:
        name_file.write(scenario_name)

    manager = P2SROManagerWithServer(
        port=psro_port,
        eval_dispatcher_port=eval_port,
        n_players=2,
        is_two_player_symmetric_zero_sum=False,
        do_external_payoff_evals_for_new_fixed_policies=True,
        games_per_external_payoff_eval=scenario.games_per_payoff_eval,
        payoff_table_exponential_average_coeff=None,
        log_dir=log_dir,
        manager_metadata={"ray_head_address": ray_head_address},
    )
    print(f"Launched P2SRO Manager with server.")

    launch_evals(scenario_name=scenario_name,
                 block=False,
                 ray_head_address=ray_head_address,
                 eval_dispatcher_port=eval_port,
                 eval_dispatcher_host='localhost')

    print(f"Launched evals")

    manager.wait_for_server_termination()
