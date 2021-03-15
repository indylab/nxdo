import logging
import os
import tempfile
import time
from typing import Dict, Type

import deepdish
import ray
from ray.rllib.utils import merge_dicts, try_import_torch
from tables.exceptions import HDF5ExtError

torch, _ = try_import_torch()
from ray.rllib.agents.dqn import DQNTrainer
import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy

from grl.utils.strategy_spec import StrategySpec
from grl.utils.common import pretty_dict_str, datetime_str, ensure_dir
from grl.rllib_tools.space_saving_logger import SpaceSavingLogger, get_trainer_logger_creator
from grl.rl_apps.scenarios.catalog import scenario_catalog
from grl.rl_apps.scenarios.psro_scenario import PSROScenario
from grl.rl_apps.scenarios.stopping_conditions import StoppingCondition

from grl.rl_apps.scenarios.ray_setup import init_ray_for_scenario

logger = logging.getLogger(__name__)


def checkpoint_dir(trainer):
    return os.path.join(trainer.logdir, "br_checkpoints")


def save_best_response_checkpoint(trainer: DQNTrainer,
                                  player: int,
                                  save_dir: str,
                                  timesteps_training_br: int,
                                  episodes_training_br: int,
                                  active_policy_num: int = None):
    policy_name = active_policy_num if active_policy_num is not None else "unclaimed"
    date_time = datetime_str()
    checkpoint_name = f"policy_{policy_name}_{date_time}.h5"
    checkpoint_path = os.path.join(save_dir, checkpoint_name)
    br_weights = trainer.get_weights([f"best_response"])["best_response"]
    br_weights = {k.replace(".", "_dot_"): v for k, v in
                  br_weights.items()}  # periods cause HDF5 NaturalNaming warnings
    ensure_dir(file_path=checkpoint_path)
    num_save_attempts = 5
    for attempt in range(num_save_attempts):
        try:
            deepdish.io.save(path=checkpoint_path, data={
                "weights": br_weights,
                "player": player,
                "policy_num": active_policy_num,
                "date_time_str": date_time,
                "seconds_since_epoch": time.time(),
                "timesteps_training_br": timesteps_training_br,
                "episodes_training_br": episodes_training_br
            })
            break
        except HDF5ExtError:
            if attempt + 1 == num_save_attempts:
                raise
            time.sleep(1.0)
    return checkpoint_path


def load_metanash_pure_strat(policy: Policy, pure_strat_spec: StrategySpec):
    pure_strat_checkpoint_path = pure_strat_spec.metadata["checkpoint_path"]

    pure_strat_checkpoint_path = pure_strat_checkpoint_path.replace(
        "/home/jb/git/grl/grl/data/12_no_limit_leduc_psro_dqn_gpu/manager_10.41.42PM_Feb-02-2021/",
        "/home/jblanier/gokuleduc/psro/manager_10.41.42PM_Feb-02-2021/")

    checkpoint_data = deepdish.io.load(path=pure_strat_checkpoint_path)
    weights = checkpoint_data["weights"]
    weights = {k.replace("_dot_", "."): v for k, v in weights.items()}
    policy.set_weights(weights=weights)
    policy.p2sro_policy_spec = pure_strat_spec


def update_all_workers_to_latest_metanash(trainer: DQNTrainer,
                                          metanash_policy_specs,
                                          metanash_weights,
                                          ):
    def _set_opponent_policy_distribution_for_one_worker(worker: RolloutWorker):
        worker.metanash_policy_specs = metanash_policy_specs
        worker.metanash_weights = metanash_weights

    trainer.workers.foreach_worker(_set_opponent_policy_distribution_for_one_worker)


class P2SROPreAndPostEpisodeCallbacks(DefaultCallbacks):

    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        # Sample new pure strategy policy weights from the metanash of the subgame population for the best response to
        # train against. For better runtime performance, consider loading new weights only every few episodes instead.
        metanash_policy: Policy = policies[f"metanash"]
        metanash_policy_specs = worker.metanash_policy_specs
        metanash_weights = worker.metanash_weights

        new_pure_strat_spec: StrategySpec = np.random.choice(a=metanash_policy_specs, p=metanash_weights)
        # noinspection PyTypeChecker
        load_metanash_pure_strat(policy=metanash_policy, pure_strat_spec=new_pure_strat_spec)


def train_poker_approx_best_response_psro(br_player,
                                          ray_head_address,
                                          scenario_name,
                                          general_trainer_config_overrrides,
                                          br_policy_config_overrides: dict,
                                          get_stopping_condition,
                                          metanash_policy_specs,
                                          metanash_weights,
                                          results_dir,
                                          print_train_results=True):
    scenario: PSROScenario = scenario_catalog.get(scenario_name=scenario_name)

    env_class = scenario.env_class
    env_config = scenario.env_config
    trainer_class = scenario.trainer_class
    policy_classes: Dict[str, Type[Policy]] = scenario.policy_classes
    p2sro = scenario.p2sro
    get_trainer_config = scenario.get_trainer_config
    psro_get_stopping_condition = scenario.psro_get_stopping_condition
    mix_metanash_with_uniform_dist_coeff = scenario.mix_metanash_with_uniform_dist_coeff

    other_player = 1 - br_player

    br_learner_name = f"new_learner_{br_player}"

    def log(message, level=logging.INFO):
        logger.log(level, f"({br_learner_name}): {message}")

    def select_policy(agent_id):
        if agent_id == br_player:
            return f"best_response"
        elif agent_id == other_player:
            return f"metanash"
        else:
            raise ValueError(f"Unknown agent id: {agent_id}")

    init_ray_for_scenario(scenario=scenario, head_address=ray_head_address, logging_level=logging.INFO)

    tmp_env = env_class(env_config=env_config)

    trainer_config = {
        "callbacks": P2SROPreAndPostEpisodeCallbacks,
        "env": env_class,
        "env_config": env_config,
        "gamma": 1.0,
        "num_gpus": 0,
        "num_workers": 0,
        "num_envs_per_worker": 1,
        "multiagent": {
            "policies_to_train": [f"best_response"],
            "policies": {
                f"metanash": (
                policy_classes["metanash"], tmp_env.observation_space, tmp_env.action_space, {"explore": scenario.allow_stochastic_best_responses}),
                f"best_response": (policy_classes["best_response"], tmp_env.observation_space, tmp_env.action_space,
                                   br_policy_config_overrides),
            },
            "policy_mapping_fn": select_policy,
        },
    }

    trainer_config = merge_dicts(trainer_config, get_trainer_config(tmp_env))
    trainer_config = merge_dicts(trainer_config, general_trainer_config_overrrides)

    # trainer_config["rollout_fragment_length"] = trainer_config["rollout_fragment_length"] // max(1, trainer_config["num_workers"] * trainer_config["num_envs_per_worker"] )

    trainer = trainer_class(config=trainer_config,
                            logger_creator=get_trainer_logger_creator(base_dir=results_dir, scenario_name="approx_br", should_log_result_fn=lambda result: result["training_iteration"] % 100 == 0))

    update_all_workers_to_latest_metanash(trainer=trainer,
                                          metanash_policy_specs=metanash_policy_specs,
                                          metanash_weights=metanash_weights)

    train_iter_count = 0

    stopping_condition: StoppingCondition = get_stopping_condition()
    max_reward = None
    while True:
        train_iter_results = trainer.train()  # do a step (or several) in the main RL loop
        train_iter_count += 1

        if print_train_results:
            train_iter_results["best_response_player"] = br_player
            # Delete verbose debugging info before printing
            if "hist_stats" in train_iter_results:
                del train_iter_results["hist_stats"]
            if "td_error" in train_iter_results["info"]["learner"][f"best_response"]:
                del train_iter_results["info"]["learner"][f"best_response"]["td_error"]
            print(pretty_dict_str(train_iter_results))

        total_timesteps_training_br = train_iter_results["timesteps_total"]
        total_episodes_training_br = train_iter_results["episodes_total"]
        br_reward_this_iter = train_iter_results["policy_reward_mean"][f"best_response"]
        if max_reward is None or br_reward_this_iter > max_reward:
            max_reward = br_reward_this_iter
        if stopping_condition.should_stop_this_iter(latest_trainer_result=train_iter_results):
            break

    trainer.cleanup()
    ray.shutdown()
    time.sleep(10)

    return max_reward
