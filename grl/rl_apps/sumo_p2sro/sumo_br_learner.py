import os
import time
import logging
import numpy as np
from typing import Dict, List
import tempfile

import deepdish

import ray
from ray.rllib.agents.sac import SACTrainer, SACTorchPolicy
from ray.rllib.utils import merge_dicts, try_import_torch
from ray.rllib.utils.torch_ops import convert_to_non_torch_type, \
    convert_to_torch_tensor
from ray.rllib.utils.typing import ModelGradients, ModelWeights, \
    TensorType, TrainerConfigDict
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.tune.logger import Logger, UnifiedLogger

import grl
from grl.p2sro.payoff_table import PayoffTableStrategySpec
from grl.utils import pretty_dict_str, datetime_str, ensure_dir
from grl.p2sro.p2sro_manager import RemoteP2SROManagerClient
from grl.p2sro.p2sro_manager.utils import get_latest_metanash_strategies, PolicySpecDistribution

from grl.rl_apps.sumo_p2sro.sumo_multi_agent_env import SumoMultiAgentEnv
from grl.rl_apps.sumo_p2sro.config import sumo_sac_params

logger = logging.getLogger(__name__)

torch, _ = try_import_torch()


def get_trainer_logger_creator(base_dir: str, env_class):
    logdir_prefix = f"{env_class.__name__}_{datetime_str()}"

    def trainer_logger_creator(config):
        """Creates a Unified logger with a default logdir prefix
        containing the agent name and the env id
        """
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        logdir = tempfile.mkdtemp(
            prefix=logdir_prefix, dir=base_dir)
        return UnifiedLogger(config, logdir)

    return trainer_logger_creator


def checkpoint_dir(trainer: SACTrainer):
    return os.path.join(trainer.logdir, "br_checkpoints")


def save_best_response_checkpoint(trainer: SACTrainer,
                                  save_dir: str,
                                  timesteps_training_br: int,
                                  episodes_training_br: int,
                                  active_policy_num: int = None):
    policy_name = active_policy_num if active_policy_num is not None else "unclaimed"
    date_time = datetime_str()
    checkpoint_name = f"policy_{policy_name}_{date_time}.h5"
    checkpoint_path = os.path.join(save_dir, checkpoint_name)
    br_weights = trainer.get_weights(["best_response"])["best_response"]
    br_weights = {k.replace(".", "_dot_"): v for k, v in br_weights.items()} # periods cause HDF5 NaturalNaming warnings
    ensure_dir(file_path=checkpoint_path)
    deepdish.io.save(path=checkpoint_path, data={
        "weights": br_weights,
        "policy_num": active_policy_num,
        "date_time_str": date_time,
        "seconds_since_epoch": time.time(),
        "timesteps_training_br": timesteps_training_br,
        "episodes_training_br": episodes_training_br
    }, )
    return checkpoint_path


def load_metanash_pure_strat(policy: SACTorchPolicy, pure_strat_spec: PayoffTableStrategySpec):
    pure_strat_checkpoint_path = pure_strat_spec.metadata["checkpoint_path"]
    checkpoint_data = deepdish.io.load(path=pure_strat_checkpoint_path)
    weights = checkpoint_data["weights"]
    weights = {k.replace("_dot_", "."): v for k, v in weights.items()}
    policy.set_weights(weights=weights)
    policy.p2sro_policy_spec = pure_strat_spec


def create_metadata_with_new_checkpoint_for_current_best_response(trainer: SACTrainer,
                                                              save_dir: str,
                                                              timesteps_training_br: int,
                                                              episodes_training_br: int,
                                                              active_policy_num: int = None,
                                                              ):
    return {
        "checkpoint_path": save_best_response_checkpoint(trainer=trainer, save_dir=save_dir,
                                                         active_policy_num=active_policy_num,
                                                         timesteps_training_br=timesteps_training_br,
                                                         episodes_training_br=episodes_training_br),
        "timesteps_training_br": timesteps_training_br,
        "episodes_training_br": episodes_training_br
    }

def set_best_response_active_policy_spec_for_all_workers(trainer: SACTrainer,
                                                         active_policy_spec: PayoffTableStrategySpec):
    def _set_p2sro_policy_spec_on_best_response_policy(worker: RolloutWorker):
        br_policy: SACTorchPolicy = worker.policy_map["best_response"]
        br_policy.p2sro_policy_spec = active_policy_spec
    trainer.workers.foreach_worker(_set_p2sro_policy_spec_on_best_response_policy)


def update_all_workers_to_latest_metanash(trainer: SACTrainer,
                                          p2sro_manager: RemoteP2SROManagerClient,
                                          active_policy_num: int):

    latest_payoff_table, active_policy_nums, fixed_policy_nums = p2sro_manager.get_copy_of_latest_data()
    latest_strategies: Dict[int, PolicySpecDistribution] = get_latest_metanash_strategies(
        payoff_table=latest_payoff_table,
        as_player=0,
        as_policy_num=active_policy_num,
        fictitious_play_iters=2000,
        mix_with_uniform_dist_coeff=0.0
    )

    if latest_strategies is None:
        opponent_policy_distribution = None
    else:
        print(f"latest payoff matrix:\n{latest_payoff_table.get_payoff_matrix_for_player(player=0)}")
        print(f"metanash: {latest_strategies[1].probabilities_for_each_strategy()}")

        # get the strategy for the opposing player, 1.
        # In a symmetric two-player game, this is the same as what player 0's strategy would be.
        opponent_policy_distribution = latest_strategies[1]

    def _set_opponent_policy_distribution_for_one_worker(worker: RolloutWorker):
        worker.opponent_policy_distribution = opponent_policy_distribution
    trainer.workers.foreach_worker(_set_opponent_policy_distribution_for_one_worker)


def sync_active_policy_br_and_metanash_with_p2sro_manager(trainer: SACTrainer,
                                                            p2sro_manager: RemoteP2SROManagerClient,
                                                            active_policy_num: int,
                                                            timesteps_training_br: int,
                                                            episodes_training_br: int):

    p2sro_manager.submit_new_active_policy_metadata(
        player=0, policy_num=active_policy_num,
        metadata_dict=create_metadata_with_new_checkpoint_for_current_best_response(
            trainer=trainer, save_dir=checkpoint_dir(trainer=trainer), timesteps_training_br=timesteps_training_br,
            episodes_training_br=episodes_training_br,
            active_policy_num=active_policy_num
        ))

    update_all_workers_to_latest_metanash(p2sro_manager=p2sro_manager, trainer=trainer,
                                          active_policy_num=active_policy_num)


# def override_best_response_sac_target_entropy(trainer: SACTrainer, new_target_entropy: float):
#     def _set_best_response_policy_target_entropy(worker: RolloutWorker):
#         br_policy: SACTorchPolicy = worker.policy_map["best_response"]
#         logger.error(f"policy.model.target_entropy: {br_policy.model.target_entropy}, {type(br_policy.model.target_entropy)}")
#         assert False
#     trainer.workers.foreach_worker(_set_best_response_policy_target_entropy)

class P2SROPreAndPostEpisodeCallbacks(DefaultCallbacks):

    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):

        # Sample new pure strategy policy weights from the metanash of the subgame population for the best response to
        # train against. For better runtime performance, consider loading new weights only every few episodes instead.
        resample_pure_strat_every_n_episodes = 1
        metanash_policy: SACTorchPolicy = policies["metanash"]
        opponent_policy_distribution: PolicySpecDistribution = worker.opponent_policy_distribution
        time_for_resample = (not hasattr(metanash_policy, "episodes_since_resample") or
                             metanash_policy.episodes_since_resample >= resample_pure_strat_every_n_episodes)
        if time_for_resample and opponent_policy_distribution is not None:
            new_pure_strat_spec: PayoffTableStrategySpec = opponent_policy_distribution.sample_policy_spec()
            # noinspection PyTypeChecker
            load_metanash_pure_strat(policy=metanash_policy, pure_strat_spec=new_pure_strat_spec)
            metanash_policy.episodes_since_resample = 1
        elif opponent_policy_distribution is not None:
            metanash_policy.episodes_since_resample += 1

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode, env_index: int,
                        **kwargs):
        super().on_episode_step(worker=worker, base_env=base_env, episode=episode, env_index=env_index, **kwargs)
        if worker.worker_index == 1 and env_index == 0:
            base_env.get_unwrapped()[0].render()

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):

        for (player, policy_name), total_reward in episode.agent_rewards.items():
            if player == 0:
                assert policy_name == "best_response"
            else:
                assert policy_name == "metanash"

            last_player_info = episode.last_info_for(agent_id=player)
            total_main_reward = last_player_info["total_main_reward"]
            total_shaping_reward = last_player_info["total_shaping_reward"]
            player_lost = int(last_player_info["lost"])

            episode.custom_metrics[f"{policy_name}_main_reward"] = total_main_reward
            episode.custom_metrics[f"{policy_name}_shaping_reward"] = total_shaping_reward
            episode.custom_metrics[f"{policy_name}_lose_rate"] = player_lost

            assert np.isclose(total_reward, total_main_reward + total_shaping_reward), \
                f"total_reward: {total_reward}, total_main_reward: {total_main_reward}, total_shaping_reward: {total_shaping_reward}"

    #     if not hasattr(worker, "p2sro_manager"):
    #         worker.p2sro_manager = RemoteP2SROManagerClient(n_players=2, port=4535, remote_server_host="127.0.0.1")
    #
    #     br_policy_spec: PayoffTableStrategySpec = worker.policy_map["best_response"].p2sro_policy_spec
    #     if br_policy_spec.pure_strat_index_for_player(player=0) == 0:
    #         # We're training policy 0 if True.
    #         # The PSRO subgame should be empty, and instead the metanash is a random neural network.
    #         # No need to report results for this.
    #         return
    #
    #     # Report payoff results for individual episodes to the p2sro manager to keep a real-time approximation of the
    #     # payoff matrix entries for (learning) active policies.
    #     policy_specs_for_each_player: List[PayoffTableStrategySpec] = [None, None]
    #     payoffs_for_each_player: List[float] = [None, None]
    #     for (player, policy_name), reward in episode.agent_rewards.items():
    #         assert policy_name in ["best_response", "metanash"]
    #         policy: SACTorchPolicy = worker.policy_map[policy_name]
    #         assert policy.p2sro_policy_spec is not None
    #         policy_specs_for_each_player[player] = policy.p2sro_policy_spec
    #         payoffs_for_each_player[player] = reward
    #     assert all(payoff is not None for payoff in payoffs_for_each_player)
    #
    #     worker.p2sro_manager.submit_empirical_payoff_result(
    #         policy_specs_for_each_player=tuple(policy_specs_for_each_player),
    #         payoffs_for_each_player=tuple(payoffs_for_each_player),
    #         games_played=1,
    #         override_all_previous_results=False)


def train_poker_sac_best_response(results_dir, print_train_results=True):

    br_learner_name = "new_learner"

    def log(message, level=logging.INFO):
        logger.log(level, f"({br_learner_name}): {message}")

    def select_policy(agent_id):
        if agent_id == 0:
            return "best_response"
        else:
            return "metanash"

    env_config = {
        "shaping_rewards": True,
        "scale_rewards": 1.0/2000.0,
    }
    tmp_env = SumoMultiAgentEnv(env_config=env_config)

    trainer_config = {
        "log_level": "DEBUG",
        "callbacks": P2SROPreAndPostEpisodeCallbacks,
        "env": SumoMultiAgentEnv,
        "env_config": env_config,
        "gamma": 0.995,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        # "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_gpus": 0.5,
        "num_workers": 25,
        "num_gpus_per_worker": 1.3/25,
        "num_envs_per_worker": 2,
        "multiagent": {
            "policies_to_train": ["best_response", "metanash"],
            "policies": {
                "metanash": (SACTorchPolicy, tmp_env.observation_space, tmp_env.action_space, {}),
                "best_response": (SACTorchPolicy, tmp_env.observation_space, tmp_env.action_space, {}),
            },
            "policy_mapping_fn": select_policy,
        },
    }
    trainer_config = merge_dicts(trainer_config, sumo_sac_params(action_space=tmp_env.action_space))

    ray.init(address='auto', _redis_password='5241590000000000')
    trainer = SACTrainer(config=trainer_config, logger_creator=get_trainer_logger_creator(base_dir=results_dir,
                                                                                          env_class=SumoMultiAgentEnv))

    p2sro_manager = RemoteP2SROManagerClient(n_players=2, port=4535, remote_server_host="127.0.0.1")
    active_policy_spec: PayoffTableStrategySpec = p2sro_manager.claim_new_active_policy_for_player(
        player=0, new_policy_metadata_dict=create_metadata_with_new_checkpoint_for_current_best_response(
            trainer=trainer, save_dir=checkpoint_dir(trainer=trainer), timesteps_training_br=0, episodes_training_br=0,
            active_policy_num=None
        ))

    active_policy_num = active_policy_spec.pure_strat_index_for_player(player=0)
    br_learner_name = f"policy {active_policy_num}"

    log(f"got policy {active_policy_num}")

    set_best_response_active_policy_spec_for_all_workers(trainer=trainer, active_policy_spec=active_policy_spec)


    def _run_once(worker: RolloutWorker):
        br_policy = worker.policy_map["best_response"]
        checkpoint_data = deepdish.io.load(path="/home/jb/git/grl/grl/data/SumoMultiAgentEnv_11.50.10PM_Nov-30-2020wddwd_ht/br_checkpoints/policy_0_12.31.18AM_Dec-01-2020.h5")
        weights = checkpoint_data["weights"]
        weights = {k.replace("_dot_", "."): v for k, v in weights.items()}
        br_policy.set_weights(weights=weights)
    trainer.workers.foreach_worker(_run_once)

    sync_active_policy_br_and_metanash_with_p2sro_manager(trainer=trainer,
                                                          p2sro_manager=p2sro_manager,
                                                          active_policy_num=active_policy_num,
                                                          timesteps_training_br=0,
                                                          episodes_training_br=0)

    # Perform main RL training loop. Stop if we reach max iters or saturate.
    # Saturation is determined by checking if we improve by a minimum amount every n iters.
    train_iter_count = 0

    sync_with_manager_every_n_train_iters = 100

    dont_do_saturation_checks_before_n_train_iters = 400000
    iters_since_saturation_checks_began = None
    check_for_saturation_every_n_train_iters = 30
    minimum_reward_improvement_otherwise_saturated = 0.05
    last_saturation_check_reward = None

    max_train_iters = 1000000

    while True:
        train_iter_results = trainer.train()  # do a step (or several) in the main RL loop
        train_iter_count += 1

        if print_train_results:
            train_iter_results["p2sro_active_policy_num"] = active_policy_num
            # Delete verbose debugging info before printing
            if "hist_stats" in train_iter_results:
                del train_iter_results["hist_stats"]
            if "td_error" in train_iter_results["info"]["learner"]["best_response"]:
                del train_iter_results["info"]["learner"]["best_response"]["td_error"]
            log(pretty_dict_str(train_iter_results))

        total_timesteps_training_br = train_iter_results["timesteps_total"]
        total_episodes_training_br = train_iter_results["episodes_total"]
        br_reward_this_iter = train_iter_results["policy_reward_mean"]["best_response"]

        if train_iter_count % sync_with_manager_every_n_train_iters == 0:
            sync_active_policy_br_and_metanash_with_p2sro_manager(trainer=trainer,
                                                                  p2sro_manager=p2sro_manager,
                                                                  active_policy_num=active_policy_num,
                                                                  timesteps_training_br=total_timesteps_training_br,
                                                                  episodes_training_br=total_episodes_training_br)

        time_to_stop_training = False

        if train_iter_count >= dont_do_saturation_checks_before_n_train_iters:
            if iters_since_saturation_checks_began is None:
                iters_since_saturation_checks_began = 0

            if iters_since_saturation_checks_began % check_for_saturation_every_n_train_iters == 0:
                if last_saturation_check_reward is not None:
                    improvement_since_last_check = br_reward_this_iter - last_saturation_check_reward
                    log(f"Improvement since last saturation check: {improvement_since_last_check}, minimum target is "
                          f"{minimum_reward_improvement_otherwise_saturated}.")
                    if improvement_since_last_check < minimum_reward_improvement_otherwise_saturated:
                        # We're no longer improving. Assume we have saturated, and stop training.
                        log(f"Improvement target not reached, stopping training if allowed.")
                        time_to_stop_training = True
                last_saturation_check_reward = br_reward_this_iter
            iters_since_saturation_checks_began += 1

        if train_iter_count >= max_train_iters:
            # Regardless of whether we've saturated, we've been training for too long, so we stop.
            log(f"Max training iters reached ({train_iter_count}). stopping training if allowed.")
            time_to_stop_training = True

        if time_to_stop_training:
            if p2sro_manager.can_active_policy_be_set_as_fixed_now(player=0, policy_num=active_policy_num):
                break
            else:
                log(f"Forcing training to continue since lower policies are still active.")

    log(f"Training stopped. Setting active policy {active_policy_num} as fixed.")

    p2sro_manager.set_active_policy_as_fixed(
        player=0, policy_num=active_policy_num,
        final_metadata_dict=create_metadata_with_new_checkpoint_for_current_best_response(
            trainer=trainer, save_dir=checkpoint_dir(trainer=trainer),
            timesteps_training_br=total_timesteps_training_br,
            episodes_training_br=total_episodes_training_br,
            active_policy_num=active_policy_num
        ))
    return active_policy_num


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    results_dir = os.path.join(os.path.dirname(grl.__file__), "data")
    print(f"results dir is {results_dir}")

    train_poker_sac_best_response(print_train_results=True, results_dir=results_dir)
    # solve inner game

