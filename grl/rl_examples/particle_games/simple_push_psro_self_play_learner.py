import os
import time
import logging
import numpy as np
from typing import Dict, List, Union
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

from grl.rl_examples.particle_games.simple_push_multi_agent_env import SimplePushMultiAgentEnv
from grl.rl_examples.particle_games.config import simple_push_sac_params_small

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
                                  br_policy_id,
                                  save_dir: str,
                                  timesteps_training_br: int,
                                  episodes_training_br: int,
                                  current_avg_br_reward: float = None,
                                  active_policy_num: int = None):
    policy_name = active_policy_num if active_policy_num is not None else "unclaimed"
    date_time = datetime_str()
    checkpoint_name = f"{br_policy_id}_{policy_name}_{date_time}.h5"
    checkpoint_path = os.path.join(save_dir, checkpoint_name)
    print(f"Saving best response checkpoint to {checkpoint_path}")
    br_weights = trainer.get_weights([br_policy_id])[br_policy_id]
    br_weights = {k.replace(".", "_dot_"): v for k, v in br_weights.items()} # periods cause HDF5 NaturalNaming warnings
    ensure_dir(file_path=checkpoint_path)
    deepdish.io.save(path=checkpoint_path, data={
        "weights": br_weights,
        "policy_num": active_policy_num,
        "date_time_str": date_time,
        "seconds_since_epoch": time.time(),
        "timesteps_training_br": timesteps_training_br,
        "episodes_training_br": episodes_training_br,
        "current_avg_br_reward": current_avg_br_reward,
    }, )
    return checkpoint_path


# class MainWorkerRenderCallback(DefaultCallbacks):
#
#
#     def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode, env_index: int,
#                         **kwargs):
#         super().on_episode_step(worker=worker, base_env=base_env, episode=episode, env_index=env_index, **kwargs)
#
#         # Debug render a single environment.
#         if worker.worker_index == 1 and env_index == 0:
#             base_env.get_unwrapped()[0].render()

def train_sac_self_play(results_dir, print_train_results=True, initial_br_weights=None):

    br_learner_name = "new learner"

    def log(message, level=logging.INFO):
        logger.log(level, f"({br_learner_name}): {message}")

    env_config = {
        "fixed_players": True
    }
    tmp_env = SimplePushMultiAgentEnv(env_config=env_config)

    trainer_config = {
        "log_level": "DEBUG",
        # "callbacks": MainWorkerRenderCallback,
        "env": SimplePushMultiAgentEnv,
        "env_config": env_config,
        "gamma": 0.9,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        # "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),

        "num_gpus": 0.1,
        "num_workers": 17,
        "num_gpus_per_worker": 0.1,
        "num_envs_per_worker": 2,

        "multiagent": {
            "policies_to_train": ["best_response_0", "best_response_1"],
            "policies": {
                "best_response_0": (SACTorchPolicy, tmp_env.observation_space, tmp_env.action_space, {}),
                "best_response_1": (SACTorchPolicy, tmp_env.observation_space, tmp_env.action_space, {}),
            },
            "policy_mapping_fn": lambda agent: "best_response_0" if agent == 0 else "best_response_1",
        },
    }
    trainer_config = merge_dicts(trainer_config, simple_push_sac_params_small(action_space=tmp_env.action_space))

    ray.init(address='auto', _redis_password='5241590000000000', ignore_reinit_error=True)

    # Create trainer
    trainer = SACTrainer(config=trainer_config, logger_creator=get_trainer_logger_creator(base_dir=results_dir,
                                                                                          env_class=SimplePushMultiAgentEnv))

    # Initialize best response weights.
    if initial_br_weights is not None:
        trainer.set_weights(initial_br_weights)

    checkpoint_iter = 0

    latest_checkpoint_paths = {}
    for br_policy_id in ["best_response_0", "best_response_1"]:
        latest_checkpoint_paths[br_policy_id] = save_best_response_checkpoint(br_policy_id=br_policy_id,
                                                               trainer=trainer, save_dir=checkpoint_dir(trainer=trainer),
                                      timesteps_training_br=0, episodes_training_br=0,
                                      current_avg_br_reward=None,
                                      active_policy_num=checkpoint_iter)

    # Perform main RL training loop.
    train_iter_count = 0

    checkpoint_every_n_train_iters = 1000

    while True:
        train_iter_results = trainer.train()  # do a step (or several) in the main RL loop
        train_iter_count += 1

        if print_train_results:
            train_iter_results["latest_checkpoint_paths"] = latest_checkpoint_paths
            # Delete verbose debugging info before printing
            if "hist_stats" in train_iter_results:
                del train_iter_results["hist_stats"]
            if "td_error" in train_iter_results["info"]["learner"]["best_response_0"]:
                del train_iter_results["info"]["learner"]["best_response_0"]["td_error"]
            if "td_error" in train_iter_results["info"]["learner"]["best_response_1"]:
                del train_iter_results["info"]["learner"]["best_response_1"]["td_error"]
            log(pretty_dict_str(train_iter_results))

        total_timesteps_training_br = train_iter_results["timesteps_total"]
        total_episodes_training_br = train_iter_results["episodes_total"]

        if train_iter_count % checkpoint_every_n_train_iters == 0:
            checkpoint_iter += 1
            for br_policy_id in ["best_response_0", "best_response_1"]:

                latest_checkpoint_paths[br_policy_id] = save_best_response_checkpoint(
                    br_policy_id=br_policy_id,
                    trainer=trainer, save_dir=checkpoint_dir(trainer=trainer),
                                              timesteps_training_br=total_timesteps_training_br,
                                              episodes_training_br=total_episodes_training_br,
                                              current_avg_br_reward=train_iter_results["policy_reward_mean"].get(br_policy_id),
                                              active_policy_num=checkpoint_iter)




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    results_dir = os.path.join(os.path.dirname(grl.__file__), "data", f"particle_psro_{datetime_str()}")
    print(f"results dir is {results_dir}")

    train_sac_self_play(
        print_train_results=True,
        results_dir=results_dir,
        initial_br_weights=None
    )

