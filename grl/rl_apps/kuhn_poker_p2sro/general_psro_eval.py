import time
from typing import Type, Dict
import os
import numpy as np
import argparse
import deepdish
import ray
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from ray.rllib.utils import merge_dicts
from ray.rllib.policy.policy import Policy
from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.sac import SACTorchPolicy, DEFAULT_CONFIG as DEFAULT_SAC_CONFIG
from grl.p2sro.eval_dispatcher.remote import RemoteEvalDispatcherClient
from grl.p2sro.payoff_table import PayoffTableStrategySpec

from grl.rl_apps.scenarios.poker import scenarios

def load_weights(policy: Policy, pure_strat_spec: PayoffTableStrategySpec):
    pure_strat_checkpoint_path = pure_strat_spec.metadata["checkpoint_path"]
    checkpoint_data = deepdish.io.load(path=pure_strat_checkpoint_path)
    weights = checkpoint_data["weights"]
    weights = {k.replace("_dot_", "."): v for k, v in weights.items()}
    policy.set_weights(weights=weights)


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
def run_poker_evaluation_loop(scenario_name: str):

    try:
        scenario = scenarios[scenario_name]
    except KeyError:
        raise NotImplementedError(f"Unknown scenario name: \'{scenario_name}\'. Existing scenarios are:\n"
                                  f"{list(scenarios.keys())}")

    env_class = scenario["env_class"]
    env_config: dict = scenario["env_config"]
    eval_policy_class: Type[Policy] = scenario["policy_classes"]["eval"]
    default_eval_port = scenario["eval_port"]
    get_trainer_config = scenario["get_trainer_config"]

    eval_dispatcher_port = os.getenv("EVAL_PORT", default_eval_port)
    eval_dispatcher_host = os.getenv("EVAL_HOST", "localhost")

    eval_dispatcher = RemoteEvalDispatcherClient(port=eval_dispatcher_port, remote_server_host=eval_dispatcher_host)

    env = env_class(env_config=env_config)
    num_players = 2

    trainer_config = get_trainer_config(action_space=env.action_space)
    trainer_config["explore"] = False

    policies = [eval_policy_class(env.observation_space,
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
                load_weights(policy=policy, pure_strat_spec=spec)

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

def launch_evals(scenario_name: str, block=True):
    try:
        scenario = scenarios[scenario_name]
    except KeyError:
        raise NotImplementedError(f"Unknown scenario name: \'{scenario_name}\'. Existing scenarios are:\n"
                                  f"{list(scenarios.keys())}")

    ray.init(log_to_driver=os.getenv("RAY_LOG_TO_DRIVER", False), address='auto', _redis_password='5241590000000000', ignore_reinit_error=False, local_mode=False)

    num_workers = scenario["num_eval_workers"]
    evaluator_refs = [run_poker_evaluation_loop.remote(scenario_name) for _ in range(num_workers)]
    if block:
        ray.wait(evaluator_refs, num_returns=num_workers)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str)
    commandline_args = parser.parse_args()

    scenario_name = commandline_args.scenario

    launch_evals(scenario_name=scenario_name)
