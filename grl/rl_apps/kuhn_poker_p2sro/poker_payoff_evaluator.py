import time
from typing import Type
import os
import numpy as np
import argparse
import deepdish
import ray
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.agents.dqn import DQNTrainer, DQNTorchPolicy, SimpleQTorchPolicy, SIMPLE_Q_DEFAULT_CONFIG

from ray.rllib.utils import merge_dicts
from ray.rllib.policy.policy import Policy
from ray.rllib.agents.sac import SACTorchPolicy, DEFAULT_CONFIG as DEFAULT_SAC_CONFIG
from grl.p2sro.eval_dispatcher.remote import RemoteEvalDispatcherClient
from grl.rl_apps.kuhn_poker_p2sro.poker_multi_agent_env import PokerMultiAgentEnv
from grl.rl_apps.kuhn_poker_p2sro.config import kuhn_sac_params, kuhn_dqn_params, leduc_dqn_params
from grl.p2sro.payoff_table import PayoffTableStrategySpec
from grl.rllib_tools.leduc_dqn.valid_actions_fcnet import LeducDQNFullyConnectedNetwork

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

@ray.remote(num_cpus=0)
def run_poker_evaluation_loop(commandline_args,
                              eval_dispatcher_port=os.getenv("EVAL_PORT", 4536),
                              eval_dispatcher_host="127.0.0.1"):

    if commandline_args.algo.lower() == 'sac':
        policy_class = SACTorchPolicy
    elif commandline_args.algo.lower() == 'dqn':
        policy_class = SimpleQTorchPolicy

    else:
        raise NotImplementedError(f"Unknown algo arg: {commandline_args.algo}")

    eval_dispatcher = RemoteEvalDispatcherClient(port=eval_dispatcher_port, remote_server_host=eval_dispatcher_host)

    env = PokerMultiAgentEnv(env_config={'version': commandline_args.env,
                                         "fixed_players": True,
                                         "append_valid_actions_mask_to_obs": commandline_args.env == "leduc_poker"})
    num_players = 2

    if commandline_args.env == "kuhn_poker":
        hyperparams = kuhn_dqn_params
    elif commandline_args.env == "leduc_poker":
        hyperparams = leduc_dqn_params
    else:
        raise NotImplementedError(f"unknown params for env: {commandline_args.env}")


    policies = [policy_class(env.observation_space,
                             env.action_space,
                             merge_dicts(SIMPLE_Q_DEFAULT_CONFIG, hyperparams(action_space=env.action_space)))
                for _ in range(num_players)]

    if commandline_args.env == "leduc_poker":
        assert isinstance(policies[0].model, LeducDQNFullyConnectedNetwork)

    while True:
        policy_specs_for_each_player, required_games_to_play = eval_dispatcher.take_eval_job()

        if policy_specs_for_each_player is None:
            time.sleep(2)
        else:
            if len(policy_specs_for_each_player) != 2:
                raise NotImplementedError(f"This evaluation code only supports two player games. "
                                          f"{len(policy_specs_for_each_player)} players were requested.")

            print(f"Got eval matchup:")
            for spec in policy_specs_for_each_player:
                print(f"spec: {spec.to_json()}")

            for policy, spec in zip(policies, policy_specs_for_each_player):
                load_weights(policy=policy, pure_strat_spec=spec)

            total_payoffs_per_player = np.zeros(shape=num_players, dtype=np.float64)

            max_reward = None
            min_reward = None

            time_since_last_output = time.time()

            for game in range(required_games_to_play):
                if game % 1000 == 0:
                    now = time.time()
                    print(f"{policy_specs_for_each_player[0].id} vs "
                          f"{policy_specs_for_each_player[1].id}: "
                          f"{game}/{required_games_to_play} games played, {now - time_since_last_output} seconds")
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='kuhn_poker', help="[kuhn_poker|leduc_poker]")
    parser.add_argument('--algo', type=str, default='dqn', help="[dqn|sac]")

    commandline_args = parser.parse_args()

    ray.init(ignore_reinit_error=False, local_mode=False)

    num_workers = 8

    evaluator_refs = [run_poker_evaluation_loop.remote(commandline_args) for _ in range(num_workers)]
    ray.wait(evaluator_refs, num_returns=num_workers)
