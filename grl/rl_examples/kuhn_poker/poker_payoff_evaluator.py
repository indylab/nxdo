import time
import numpy as np

import ray
from ray.rllib.utils import merge_dicts

from ray.rllib.agents.sac import SACTorchPolicy, DEFAULT_CONFIG as DEFAULT_SAC_CONFIG
from grl.eval_dispatcher.remote import RemoteEvalDispatcherClient
from grl.rl_examples.kuhn_poker.poker_multi_agent_env import PokerMultiAgentEnv
from grl.rl_examples.kuhn_poker.config import kuhn_sac_params


def run_poker_evaluation_loop(poker_game="kuhn_poker", eval_dispatcher_port=4536, eval_dispatcher_host="127.0.0.1"):
    ray.init(ignore_reinit_error=True, local_mode=True)

    eval_dispatcher = RemoteEvalDispatcherClient(port=eval_dispatcher_port, remote_server_host=eval_dispatcher_host)

    env = PokerMultiAgentEnv(env_config={'version': poker_game})
    num_players = 2

    policies = [SACTorchPolicy(env.observation_space, env.action_space, merge_dicts(DEFAULT_SAC_CONFIG, kuhn_sac_params)) for _ in range(num_players)]

    while True:
        policy_specs_for_each_player, required_games_to_play = eval_dispatcher.take_eval_job()
        if policy_specs_for_each_player is None:
            time.sleep(2)
        else:
            if len(policy_specs_for_each_player) != 2:
                raise NotImplemented("This evaluation code only supports two player games.")

            total_payoffs_per_player = np.zeros(shape=num_players, dtype=np.float64)

            max_reward = None
            min_reward = None

            time_since_last_output = time.time()

            for game in range(required_games_to_play):
                if game % 10 == 0:
                    now = time.time()
                    print(f"{policy_specs_for_each_player[0].id} vs "
                          f"{policy_specs_for_each_player[1].id}: "
                          f"{game}/{required_games_to_play} games played, {now - time_since_last_output} seconds")
                    time_since_last_output = now

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
                    assert len(obs) == 1
                    acting_player, acting_agent_observation = list(obs.items())[0]

                    action_index, new_policy_state, action_info = policies[acting_player].compute_single_action(
                        obs=obs, state=policy_states[acting_player])

                    policy_states[acting_player] = new_policy_state

                    obs, rewards, dones, infos = env.step(action_dict={acting_player: action_index})

                    for player in range(num_players):
                        payoff_so_far = payoffs_per_player_this_episode[player]
                        payoffs_per_player_this_episode[player] = payoff_so_far + rewards.get(player, 0.0)

                total_payoffs_per_player += payoffs_per_player_this_episode

                if max_reward is None or max(payoffs_per_player_this_episode) > max_reward:
                    max_reward = max(payoffs_per_player_this_episode)
                if min_reward is None or min(payoffs_per_player_this_episode) < min_reward:
                    min_reward = min(payoffs_per_player_this_episode)

            payoffs_per_player = total_payoffs_per_player / required_games_to_play

            print(f"payoffs per player: {payoffs_per_player}")

            eval_dispatcher.submit_eval_job_result(
                policy_specs_for_each_player_tuple=policy_specs_for_each_player,
                payoffs_for_each_player=payoffs_per_player,
                games_played=required_games_to_play
            )


if __name__ == '__main__':
    run_poker_evaluation_loop()