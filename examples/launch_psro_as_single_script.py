import argparse
import logging
import subprocess

import ray

import grl.rl_apps.psro.general_psro_eval
from grl.rl_apps import GRL_SEED
from grl.rl_apps.psro.general_psro_manager import launch_manager
from grl.rl_apps.scenarios.catalog import scenario_catalog
from grl.rl_apps.scenarios.psro_scenario import PSROScenario
from grl.utils.port_listings import establish_new_server_port_for_service

"""
Launches PSRO manager/eval processes along with BR's for players 0 and 1 in a single shell.

Most testing has been done with BR workers launched as separate scripts for better readability/debugging.
BR's connect to the manager via gRPC, so they can be launched on separate devices if needed.

Example usage:
python launch_psro_as_single_script.py --scenario kuhn_psro_dqn

"""

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str)
    parser.add_argument('--psro_port', type=int, required=False, default=None)
    parser.add_argument('--eval_port', type=int, required=False, default=None)
    parser.add_argument('--br_method', type=str, required=False, default='standard')
    parser.add_argument('--use_prev_brs', default=False, action='store_true')
    commandline_args = parser.parse_args()

    scenario_name = commandline_args.scenario
    scenario: PSROScenario = scenario_catalog.get(scenario_name=scenario_name)

    psro_port = commandline_args.psro_port
    if psro_port is None:
        psro_port = establish_new_server_port_for_service(service_name=f"seed_{GRL_SEED}_{scenario.name}")

    eval_port = commandline_args.eval_port
    if eval_port is None:
        eval_port = establish_new_server_port_for_service(service_name=f"seed_{GRL_SEED}_{scenario.name}_evals")

    manager = launch_manager(scenario=scenario, psro_port=psro_port, eval_port=eval_port, block=False,
                             include_evals=False)

    br_method = commandline_args.br_method
    use_prev_brs = commandline_args.use_prev_brs
    if br_method == "standard":
        # Ray could also be used to launch these BR processes,
        # but it isn't extensively tested in terms of proper actor cleanup.
        import grl.rl_apps.psro.general_psro_br

        br_script_path = grl.rl_apps.psro.general_psro_br.__file__
        for player in range(2):
            subprocess.Popen(["python", br_script_path, "--scenario", scenario_name, "--player", str(player)])
    else:
        raise NotImplemented(f"unknown br method: {br_method}")

    # Also launch PSRO payoff table empirical evals in a separate process.
    subprocess.Popen(["python", grl.rl_apps.psro.general_psro_eval.__file__,
                      "--scenario", scenario_name,
                      "--ray_head", manager.manager_metadata["ray_head_address"]])

    manager.wait_for_server_termination()
    ray.shutdown()
