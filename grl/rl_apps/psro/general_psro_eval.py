import argparse
import logging
import os
import time
from typing import Type

import deepdish
import numpy as np
import ray
from ray.rllib.agents.trainer import with_common_config
from ray.rllib.policy.policy import Policy

from grl.p2sro.eval_dispatcher.remote import RemoteEvalDispatcherClient
from grl.rl_apps.scenarios.catalog import scenario_catalog
from grl.rl_apps.scenarios.psro_scenario import PSROScenario
from grl.rl_apps.scenarios.ray_setup import init_ray_for_scenario
from grl.utils.strategy_spec import StrategySpec
from grl.utils.port_listings import get_client_port_for_service
from grl.rl_apps import GRL_SEED
from grl.rllib_tools.policy_checkpoints import load_pure_strat


def run_episode(env, policies_for_each_player) -> np.ndarray:
    num_players = len(policies_for_each_player)

    obs = env.reset()
    dones = {}
    game_length = 0
    policy_states = [None] * num_players

    payoffs_per_player_this_episode = np.zeros(shape=num_players, dtype=np.float64)
    while True:
        if "__all__" in dones:
            if dones["__all__"]:
                break
        game_length += 1

        action_dict = {}
        for player in range(num_players):
            if player in obs:
                action_index, new_policy_state, action_info = policies_for_each_player[player].compute_single_action(
                    obs=obs[player], state=policy_states[player])
                policy_states[player] = new_policy_state
                action_dict[player] = action_index

        obs, rewards, dones, infos = env.step(action_dict=action_dict)

        for player in range(num_players):
            payoff_so_far = payoffs_per_player_this_episode[player]
            payoffs_per_player_this_episode[player] = payoff_so_far + rewards.get(player, 0.0)

    return payoffs_per_player_this_episode


@ray.remote(num_cpus=0, num_gpus=0)
def run_poker_evaluation_loop(scenario_name: str, eval_dispatcher_port: int, eval_dispatcher_host: str):
    scenario: PSROScenario = scenario_catalog.get(scenario_name=scenario_name)

    eval_dispatcher = RemoteEvalDispatcherClient(port=eval_dispatcher_port, remote_server_host=eval_dispatcher_host)

    env = scenario.env_class(env_config=scenario.env_config)
    num_players = 2

    trainer_config = scenario.get_trainer_config(env)
    trainer_config["explore"] = False

    policies = [scenario.policy_classes["eval"](env.observation_space,
                                                env.action_space,
                                                with_common_config(trainer_config))
                for _ in range(num_players)]

    while True:
        policy_specs_for_each_player, required_games_to_play = eval_dispatcher.take_eval_job()

        if policy_specs_for_each_player is None:
            time.sleep(2)
        else:
            if len(policy_specs_for_each_player) != 2:
                raise NotImplementedError(f"This evaluation code only supports two player games. "
                                          f"{len(policy_specs_for_each_player)} players were requested.")

            # print(f"Got eval matchup:")
            # for spec in policy_specs_for_each_player:
            #     print(f"spec: {spec.to_json()}")

            for policy, spec in zip(policies, policy_specs_for_each_player):
                load_pure_strat(policy=policy, pure_strat_spec=spec)

            total_payoffs_per_player = np.zeros(shape=num_players, dtype=np.float64)

            max_reward = None
            min_reward = None

            time_since_last_output = time.time()

            for game in range(required_games_to_play):
                if game % 1000 == 0:
                    now = time.time()
                    # print(f"{policy_specs_for_each_player[0].id} vs "
                    #       f"{policy_specs_for_each_player[1].id}: "
                    #       f"{game}/{required_games_to_play} games played, {now - time_since_last_output} seconds")
                    time_since_last_output = now

                payoffs_per_player_this_episode = run_episode(env=env, policies_for_each_player=policies)

                total_payoffs_per_player += payoffs_per_player_this_episode

                if max_reward is None or max(payoffs_per_player_this_episode) > max_reward:
                    max_reward = max(payoffs_per_player_this_episode)
                if min_reward is None or min(payoffs_per_player_this_episode) < min_reward:
                    min_reward = min(payoffs_per_player_this_episode)

            payoffs_per_player = total_payoffs_per_player / required_games_to_play

            print(f"payoffs per player:"
                  f"{policy_specs_for_each_player[0].id} vs "
                  f"{policy_specs_for_each_player[1].id}: "
                  f"{payoffs_per_player}")

            eval_dispatcher.submit_eval_job_result(
                policy_specs_for_each_player_tuple=policy_specs_for_each_player,
                payoffs_for_each_player=payoffs_per_player,
                games_played=required_games_to_play
            )


def launch_evals(scenario_name: str,
                 eval_dispatcher_port: int,
                 eval_dispatcher_host: str,
                 block=True,
                 ray_head_address=None):
    scenario: PSROScenario = scenario_catalog.get(scenario_name=scenario_name)

    init_ray_for_scenario(scenario=scenario, head_address=ray_head_address, logging_level=logging.INFO)

    num_workers = scenario.num_eval_workers
    evaluator_refs = [run_poker_evaluation_loop.remote(scenario_name, eval_dispatcher_port, eval_dispatcher_host)
                      for _ in range(num_workers)]
    if block:
        ray.wait(evaluator_refs, num_returns=num_workers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str)
    parser.add_argument('--eval_port', type=int, required=False, default=None)
    parser.add_argument('--eval_host', type=str, required=False, default='localhost')
    commandline_args = parser.parse_args()

    scenario_name = commandline_args.scenario
    eval_port = commandline_args.eval_port
    if eval_port is None:
        eval_port = get_client_port_for_service(service_name=f"seed_{GRL_SEED}_{scenario_name}_evals")

    launch_evals(scenario_name=scenario_name,
                 eval_dispatcher_port=eval_port,
                 eval_dispatcher_host=commandline_args.eval_host)
