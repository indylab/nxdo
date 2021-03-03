from ray.rllib.utils import try_import_torch

torch, _ = try_import_torch()

import os
import logging
import argparse

import grl
from grl.algos.xfdo.xfdo_manager.remote import XFDOManagerWithServer
from grl.algos.xfdo.xfdo_manager.manager import SolveRestrictedGame
from grl.utils.common import datetime_str, ensure_dir
from grl.rl_apps.scenarios.catalog import scenario_catalog
from grl.rl_apps.scenarios.nxdo_scenario import NXDOScenario
from grl.rl_apps.scenarios.ray_setup import init_ray_for_scenario
from grl.utils.port_listings import establish_new_server_port_for_service
from grl.rl_apps import GRL_SEED

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str)
    parser.add_argument('--nxdo_port', type=int, required=False, default=None)
    commandline_args = parser.parse_args()

    scenario_name = commandline_args.scenario
    scenario: NXDOScenario = scenario_catalog.get(scenario_name=scenario_name)

    nxdo_port = commandline_args.nxdo_port
    if nxdo_port is None:
        nxdo_port = establish_new_server_port_for_service(service_name=f"seed_{GRL_SEED}_{scenario.name}")

    ray_head_address = init_ray_for_scenario(scenario=scenario, head_address=None, logging_level=logging.INFO)

    solve_restricted_game: SolveRestrictedGame = scenario.get_restricted_game_solver(scenario)

    log_dir = os.path.join(os.path.dirname(grl.__file__), "data", scenario_name, f"manager_{datetime_str()}")

    name_file_path = os.path.join(log_dir, "scenario_name.txt")
    ensure_dir(name_file_path)
    with open(name_file_path, "w+") as name_file:
        name_file.write(scenario_name)

    manager = XFDOManagerWithServer(
        solve_restricted_game=solve_restricted_game,
        n_players=2,
        log_dir=os.path.join(os.path.dirname(grl.__file__), "data", scenario_name, f"manager_{datetime_str()}"),
        port=nxdo_port,
        manager_metadata={"ray_head_address": ray_head_address},
    )

    manager.wait_for_server_termination()
