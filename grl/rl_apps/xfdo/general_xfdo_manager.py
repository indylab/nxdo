import ray
from ray.rllib.utils import merge_dicts, try_import_torch
torch, _ = try_import_torch()


import os
import logging
import argparse

import grl
from grl.xfdo.xfdo_manager.remote import XFDOManagerWithServer
from grl.xfdo.xfdo_manager.manager import SolveRestrictedGame
from grl.utils import datetime_str, ensure_dir
from grl.rl_apps.scenarios.poker import scenarios


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

    default_xfdo_port = scenario["xfdo_port"]
    solve_restricted_game: SolveRestrictedGame = scenario["get_restricted_game_solver"](scenario)

    log_dir = os.path.join(os.path.dirname(grl.__file__), "data", scenario_name, f"manager_{datetime_str()}")

    name_file_path = os.path.join(log_dir, "scenario_name.txt")
    ensure_dir(name_file_path)
    with open(name_file_path, "w+") as name_file:
        name_file.write(scenario_name)

    manager = XFDOManagerWithServer(
        solve_restricted_game=solve_restricted_game,
        n_players=2,
        log_dir=os.path.join(os.path.dirname(grl.__file__), "data", scenario_name, f"manager_{datetime_str()}"),
        port=os.getenv("XFDO_PORT", default_xfdo_port))

    manager.wait_for_server_termination()