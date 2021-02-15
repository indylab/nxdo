import os
import time
import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Callable, Type, Dict, Union
import tempfile
import argparse

from gym.spaces import Space
import copy
import deepdish

import ray
from ray.rllib.utils import merge_dicts, try_import_torch
torch, _ = try_import_torch()

from ray.rllib import SampleBatch, Policy
from ray.rllib.agents import Trainer
from ray.rllib.utils.torch_ops import convert_to_non_torch_type, \
    convert_to_torch_tensor
from ray.rllib.utils.typing import ModelGradients, ModelWeights, \
    TensorType, TrainerConfigDict, AgentID, PolicyID
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.tune.logger import Logger, UnifiedLogger
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID, \
    MultiAgentBatch
from ray.rllib.agents.dqn.dqn_tf_policy import PRIO_WEIGHTS
import grl
from grl.utils import pretty_dict_str, datetime_str, ensure_dir, copy_attributes

from grl.nfsp_rllib.nfsp import get_store_to_avg_policy_buffer_fn
from grl.rl_apps.nfsp.openspiel_utils import nfsp_measure_exploitability_nonlstm
from grl.rllib_tools.space_saving_logger import SpaceSavingLogger
from grl.rl_apps.scenarios.poker import scenarios
from grl.rl_apps.scenarios.ray_setup import init_ray_for_scenario
from grl.rl_apps.scenarios.stopping_conditions import StoppingCondition
from grl.p2sro.payoff_table import PayoffTableStrategySpec

logger = logging.getLogger(__name__)


def load_pure_strat(policy: Policy, pure_strat_spec, checkpoint_path: str = None):
    assert pure_strat_spec is None or checkpoint_path is None, "can only pass one or the other"
    if checkpoint_path is None:
        if hasattr(policy, "policy_spec") and pure_strat_spec == policy.policy_spec:
            return
        pure_strat_checkpoint_path = pure_strat_spec.metadata["checkpoint_path"]
    else:
        pure_strat_checkpoint_path = checkpoint_path

    pure_strat_checkpoint_path = pure_strat_checkpoint_path.replace("/home/jb/git/grl/grl/data/12_no_limit_leduc_nfsp_dqn_gpu_sparse_10.44.52PM_Feb-02-20213spj98kj/", "/home/jblanier/gokuleduc/nfsp/12_no_limit_leduc_nfsp_dqn_gpu_sparse_10.44.52PM_Feb-02-20213spj98kj/")

    checkpoint_data = deepdish.io.load(path=pure_strat_checkpoint_path)
    weights = checkpoint_data["weights"]
    weights = {k.replace("_dot_", "."): v for k, v in weights.items()}
    policy.set_weights(weights=weights)
    policy.policy_spec = pure_strat_spec

def get_trainer_logger_creator(base_dir: str, scenario_name: str):
    logdir_prefix = f"{scenario_name}_sparse_{datetime_str()}"

    def trainer_logger_creator(config):
        """Creates a Unified logger with a default logdir prefix
        containing the agent name and the env id
        """
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        logdir = tempfile.mkdtemp(
            prefix=logdir_prefix, dir=base_dir)

        def _should_log(result: dict) -> bool:
            return ("z_avg_policy_exploitability" in result or
                    result["training_iteration"] % 1000 == 0)

        return SpaceSavingLogger(config=config, logdir=logdir, should_log_result_fn=_should_log, print_log_dir=False)

    return trainer_logger_creator


def checkpoint_dir(trainer: Trainer):
    return os.path.join(trainer.logdir, "avg_policy_checkpoints")

def spec_checkpoint_dir(trainer: Trainer):
    return os.path.join(trainer.logdir, "avg_policy_checkpoint_specs")


def train_poker_approx_best_response_nfsp(br_player,
                             ray_head_address,
                             scenario,
                             general_trainer_config_overrrides,
                             br_policy_config_overrides,
                             get_stopping_condition,
                             avg_policy_specs_for_players: Dict[int, PayoffTableStrategySpec],
                             results_dir: str,
                             print_train_results: bool = True):

    env_class = scenario["env_class"]
    env_config = scenario["env_config"]
    trainer_class = scenario["trainer_class"]
    avg_trainer_class = scenario["avg_trainer_class"]
    policy_classes: Dict[str, Type[Policy]] = scenario["policy_classes"]
    anticipatory_param: float = scenario["anticipatory_param"]
    get_trainer_config = scenario["get_trainer_config"]
    get_avg_trainer_config = scenario["get_avg_trainer_config"]
    calculate_openspiel_metanash: bool = scenario["calculate_openspiel_metanash"]
    calc_metanash_every_n_iters: int = scenario["calc_metanash_every_n_iters"]
    checkpoint_every_n_iters: Union[int, None] = scenario["checkpoint_every_n_iters"]
    nfsp_get_stopping_condition = scenario["nfsp_get_stopping_condition"]

    init_ray_for_scenario(scenario=scenario, head_address=ray_head_address, logging_level=logging.INFO)

    def log(message, level=logging.INFO):
        logger.log(level, message)

    def select_policy(agent_id):
        random_sample = np.random.random()
        if agent_id == br_player:
            return "best_response"
        else:
            return f"average_policy"


    def assert_not_called(agent_id):
        assert False, "This function should never be called."

    tmp_env = env_class(env_config=env_config)
    open_spiel_env_config = tmp_env.open_spiel_env_config if calculate_openspiel_metanash else None

    avg_policy_model_config = get_trainer_config(action_space=tmp_env.action_space)["model"]

    br_trainer_config = {
        "log_level": "DEBUG",
        # "callbacks": None,
        "env": env_class,
        "env_config": env_config,
        "gamma": 1.0,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        # "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_gpus": 0.0,
        "num_workers": 0,
        "num_gpus_per_worker": 0.0,
        "num_envs_per_worker": 1,
        "multiagent": {
            "policies_to_train": ["best_response"],
            "policies": {
                "average_policy": (policy_classes["average_policy"], tmp_env.observation_space, tmp_env.action_space, {
                    "model": avg_policy_model_config,
                    "explore": False,
                }),
                "best_response": (policy_classes["best_response"], tmp_env.observation_space, tmp_env.action_space, br_policy_config_overrides),
            },
            "policy_mapping_fn": select_policy,
        },
    }
    br_trainer_config = merge_dicts(br_trainer_config, get_trainer_config(tmp_env.action_space))

    br_trainer_config = merge_dicts(br_trainer_config, general_trainer_config_overrrides)

    br_trainer = trainer_class(config=br_trainer_config,
                               logger_creator=get_trainer_logger_creator(base_dir=results_dir, scenario_name="approx_br"))

    def _set_avg_policy(worker: RolloutWorker):
        avg_policy = worker.policy_map["average_policy"]
        load_pure_strat(policy=avg_policy, pure_strat_spec=avg_policy_specs_for_players[1 - br_player])
    br_trainer.workers.foreach_worker(_set_avg_policy)

    br_trainer.latest_avg_trainer_result = None
    train_iter_count = 0

    stopping_condition: StoppingCondition = get_stopping_condition()

    max_reward = None
    while True:
        train_iter_results = br_trainer.train()  # do a step (or several) in the main RL loop
        br_reward_this_iter = train_iter_results["policy_reward_mean"][f"best_response"]

        if max_reward is None or br_reward_this_iter > max_reward:
            max_reward = br_reward_this_iter

        train_iter_count += 1
        if print_train_results:
            # Delete verbose debugging info before printing
            if "hist_stats" in train_iter_results:
                del train_iter_results["hist_stats"]
            if "td_error" in train_iter_results["info"]["learner"]["best_response"]:
                del train_iter_results["info"]["learner"]["best_response"]["td_error"]
            print(pretty_dict_str(train_iter_results))
            log(f"Trainer logdir is {br_trainer.logdir}")

        if stopping_condition.should_stop_this_iter(latest_trainer_result=train_iter_results):
            print("stopping condition met.")
            break

    return max_reward
