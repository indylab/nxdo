import argparse
import logging

import ray
import reap

from grl.rl_apps import GRL_SEED
from grl.rl_apps.nxdo.general_nxdo_manager import launch_manager
from grl.rl_apps.scenarios.catalog import scenario_catalog
from grl.rl_apps.scenarios.nxdo_scenario import NXDOScenario
from grl.utils.port_listings import establish_new_server_port_for_service

"""
Launches NXDO manager along with BR's for players 0 and 1 in a single shell.

Most testing has been done with BR workers launched as separate scripts for better readability/debugging.
BR's connect to the manager via gRPC, so they can be launched on separate devices if needed.

Example usage:
python launch_nxdo_as_single_script.py --scenario 1_step_kuhn_nxdo_dqn_nfsp

"""

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str)
    parser.add_argument('--nxdo_port', type=int, required=False, default=None)
    parser.add_argument('--br_method', type=str, required=False, default='standard')
    parser.add_argument('--use_prev_brs', default=False, action='store_true')

    commandline_args = parser.parse_args()

    scenario_name = commandline_args.scenario
    scenario: NXDOScenario = scenario_catalog.get(scenario_name=scenario_name)

    nxdo_port = commandline_args.nxdo_port
    if nxdo_port is None:
        nxdo_port = establish_new_server_port_for_service(service_name=f"seed_{GRL_SEED}_{scenario.name}")

    manager = launch_manager(scenario=scenario, nxdo_port=nxdo_port, block=False)

    br_method = commandline_args.br_method
    use_prev_brs = commandline_args.use_prev_brs
    if br_method == "standard":
        # Ray could also be used to launch these BR processes,
        # but it isn't extensively tested in terms of proper actor cleanup.
        import grl.rl_apps.nxdo.general_nxdo_br

        br_script_path = grl.rl_apps.nxdo.general_nxdo_br.__file__
        for player in range(2):
            reap.Popen(["python", br_script_path, "--scenario", scenario_name, "--player", str(player)])
    else:
        raise NotImplemented(f"unknown br method: {br_method}")

    manager.wait_for_server_termination()
    ray.shutdown()
