import logging
import time

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from grl.rl_apps.kuhn_poker_p2sro.poker_multi_agent_env import PokerMultiAgentEnv

logger = logging.getLogger(__name__)


def eval_policy_matchup(get_policy_fn_a, get_policy_fn_b, env: MultiAgentEnv, env_config, games_per_matchup):
    resample_policy_fn_a = False
    if isinstance(get_policy_fn_a, tuple):
        get_policy_fn_a, resample_policy_fn_a = get_policy_fn_a

    policy_a_name, policy_a_get_action_index = get_policy_fn_a(env.__class__, env_config)

    resample_policy_fn_b = False
    if isinstance(get_policy_fn_b, tuple):
        get_policy_fn_b, resample_policy_fn_b = get_policy_fn_b

    policy_b_name, policy_b_get_action_index = get_policy_fn_b(env.__class__, env_config)
    policy_funcs = [policy_a_get_action_index, policy_b_get_action_index]

    policy_a_state = None
    policy_b_state = None
    policy_states = [policy_a_state, policy_b_state]

    def policy_index(agent_id):
        if agent_id == 1:
            return 0
        else:
            return 1

    policy_a_total_payoff = 0
    ties = 0

    max_reward = None
    min_reward = None

    time_since_last_output = time.time()

    for game in range(games_per_matchup):
        if game % 10 == 0:
            now = time.time()
            logger.debug(f"{policy_a_name} vs {policy_b_name}: {game}/{games_per_matchup} games played, {now - time_since_last_output} seconds")
            time_since_last_output = now

        if resample_policy_fn_a:
            policy_a_get_action_index(None, None, resample=True)

        if resample_policy_fn_b:
            policy_b_get_action_index(None, None, resample=True)

        obs = env.reset()
        dones = {}
        infos = {}
        game_length = 0

        player_a_total_game_reward = 0.0
        while True:
            if "__all__" in dones:
                if dones["__all__"]:
                    break
            game_length += 1
            assert len(obs) == 1
            acting_agent_id, acting_agent_observation = list(obs.items())[0]
            acting_policy_fn = policy_funcs[policy_index(acting_agent_id)]
            acting_policy_state = policy_states[policy_index(acting_agent_id)]

            action_index, new_policy_state = acting_policy_fn(acting_agent_observation, acting_policy_state)
            policy_states[policy_index(acting_agent_id)] = new_policy_state

            obs, rewards, dones, infos = env.step(action_dict={acting_agent_id: action_index})
            player_a_total_game_reward += rewards.get(1, 0.0)

        player_a_won = infos[1]['game_result'] == 'won'
        tied = infos[1]['game_result'] == 'tied'

        policy_a_total_payoff += player_a_total_game_reward
        if player_a_total_game_reward > 0:
            assert player_a_won, f"player_a_total_game_reward: {player_a_total_game_reward}"

        if tied:
            ties += 1

        if max_reward is None or player_a_total_game_reward > max_reward:
            max_reward = player_a_total_game_reward
        if min_reward is None or player_a_total_game_reward < min_reward:
            min_reward = player_a_total_game_reward

    policy_a_avg_payoff = policy_a_total_payoff / games_per_matchup
    tie_percentage = ties / games_per_matchup

    logger.info(f"max reward for player a: {max_reward}, min reward {min_reward}")
    logger.info(f"avg payoff: {policy_a_avg_payoff}")
    return policy_a_avg_payoff, tie_percentage


def make_get_policy_fn(model_weights_path, model_config, policy_name, policy_class):

    def get_policy_fn(env_class, env_config):
        from ray.rllib.agents.trainer import with_common_config
        import ray
        import deepdish

        ray.init(ignore_reinit_error=True, local_mode=True)

        tmp_env = env_class(env_config=env_config)
        policy = policy_class(
            obs_space=tmp_env.observation_space,
            action_space=tmp_env.action_space,
            config=with_common_config({
                'model': model_config,
                'env': PokerMultiAgentEnv,
                'env_config': env_config
            })
        )
        checkpoint_data = deepdish.io.load(path=model_weights_path)
        policy.set_weights(checkpoint_data["weights"])

        def policy_fn(observation, policy_state=None):
            if policy_state is None:
                policy_state = policy.get_initial_state()

            current_player_perspective_action_index, policy_state, _ = policy.compute_single_action(
                obs=observation,
                state=policy_state)

            return current_player_perspective_action_index, policy_state

        # policy name must be unique
        return policy_name, policy_fn

    return get_policy_fn