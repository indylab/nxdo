import argparse
import logging
import os

import grl
from grl.algos.nxdo.nxdo_manager.manager import SolveRestrictedGame
from grl.algos.nxdo.nxdo_manager.remote import NXDOManagerWithServer
from grl.rl_apps import GRL_SEED
from grl.rl_apps.scenarios.catalog import scenario_catalog
from grl.rl_apps.scenarios.nxdo_scenario import NXDOScenario
from grl.rl_apps.scenarios.ray_setup import init_ray_for_scenario
from grl.utils.common import datetime_str, ensure_dir
from grl.utils.port_listings import establish_new_server_port_for_service


def launch_manager(scenario: NXDOScenario, nxdo_port: int, block: bool = True) -> NXDOManagerWithServer:
    if not isinstance(scenario, NXDOScenario):
        raise TypeError(f"Only instances of {NXDOScenario} can be used here. {scenario.name} is a {type(scenario)}.")

    ray_head_address = init_ray_for_scenario(scenario=scenario, head_address=None, logging_level=logging.INFO)

    solve_restricted_game: SolveRestrictedGame = scenario.get_restricted_game_solver(scenario)

    log_dir = os.path.join(os.path.dirname(grl.__file__), "data", scenario.name, f"manager_{datetime_str()}")

    name_file_path = os.path.join(log_dir, "scenario_name.txt")
    ensure_dir(name_file_path)
    with open(name_file_path, "w+") as name_file:
        name_file.write(scenario.name)

    manager = NXDOManagerWithServer(
        solve_restricted_game=solve_restricted_game,
        n_players=2,
        log_dir=os.path.join(os.path.dirname(grl.__file__), "data", scenario.name, f"manager_{datetime_str()}"),
        port=nxdo_port,
        manager_metadata={"ray_head_address": ray_head_address},
    )

    if block:
        manager.wait_for_server_termination()

    return manager


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

    launch_manager(scenario=scenario, nxdo_port=nxdo_port, block=True)
