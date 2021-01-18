import os
import time
import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Callable, Type, Dict
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
from grl.rl_apps.scenarios.stopping_conditions import StoppingCondition

logger = logging.getLogger(__name__)


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

        return SpaceSavingLogger(config=config, logdir=logdir, should_log_result_fn=_should_log)

    return trainer_logger_creator


def checkpoint_dir(trainer: Trainer):
    return os.path.join(trainer.logdir, "br_checkpoints")


def train_poker_off_policy_rl_nfsp(results_dir: str,
                                   scenario_name: str,
                                   print_train_results: bool = True):

    try:
        scenario = scenarios[scenario_name]
    except KeyError:
        raise NotImplementedError(f"Unknown scenario name: \'{scenario_name}\'. Existing scenarios are:\n"
                                  f"{list(scenarios.keys())}")
    env_class = scenario["env_class"]
    env_config = scenario["env_config"]
    trainer_class = scenario["trainer_class"]
    avg_trainer_class = scenario["avg_trainer_class"]
    policy_classes: Dict[str, Type[Policy]] = scenario["policy_classes"]
    anticipatory_param: float = scenario["anticipatory_param"]
    get_trainer_config = scenario["get_trainer_config"]
    calculate_openspiel_metanash: bool = scenario["calculate_openspiel_metanash"]
    calc_metanash_every_n_iters: int = scenario["calc_metanash_every_n_iters"]
    nfsp_get_stopping_condition = scenario["nfsp_get_stopping_condition"]

    ray.init(ignore_reinit_error=True, local_mode=False)

    def log(message, level=logging.INFO):
        logger.log(level, message)

    def select_policy(agent_id):
        random_sample = np.random.random()
        if agent_id == 0:
            if random_sample < anticipatory_param:
                return "best_response_0"
            return "average_policy_0"
        elif agent_id == 1:
            if random_sample < anticipatory_param:
                return "best_response_1"
            return "average_policy_1"
        else:
            raise ValueError(f"unexpected agent_id: {agent_id}")

    def assert_not_called(agent_id):
        assert False, "This function should never be called."

    tmp_env = env_class(env_config=env_config)

    avg_policy_model_config = get_trainer_config(action_space=tmp_env.action_space)["model"]

    avg_trainer = avg_trainer_class(config={
        "log_level": "DEBUG",
        "framework": "torch",
        "env": env_class,
        "env_config": env_config,
        "num_gpus": 0.0,
        "num_gpus_per_worker": 0.0,
        "num_workers": 0,
        "num_envs_per_worker": 1,
        "multiagent": {
            "policies_to_train": ["average_policy_0", "average_policy_1"],
            "policies": {
                "average_policy_0": (policy_classes["average_policy"], tmp_env.observation_space, tmp_env.action_space, {
                    "model": avg_policy_model_config
                }),
                "average_policy_1": (policy_classes["average_policy"], tmp_env.observation_space, tmp_env.action_space, {
                    "model": avg_policy_model_config
                }),
            },
            "policy_mapping_fn": assert_not_called,
        },

    }, logger_creator=get_trainer_logger_creator(base_dir=results_dir, scenario_name=scenario_name))

    store_to_avg_policy_buffer = get_store_to_avg_policy_buffer_fn(nfsp_trainer=avg_trainer)

    class NFSPBestResponseCallbacks(DefaultCallbacks):

        def on_postprocess_trajectory(self, *, worker: "RolloutWorker", episode: MultiAgentEpisode, agent_id: AgentID,
                                      policy_id: PolicyID, policies: Dict[PolicyID, Policy],
                                      postprocessed_batch: SampleBatch,
                                      original_batches: Dict[Any, Tuple[Policy, SampleBatch]],
                                      **kwargs):
            super().on_postprocess_trajectory(worker=worker, episode=episode, agent_id=agent_id, policy_id=policy_id,
                                              policies=policies, postprocessed_batch=postprocessed_batch,
                                              original_batches=original_batches, **kwargs)

            postprocessed_batch.data["source_policy"] = [policy_id] * len(postprocessed_batch.data["rewards"])

            # All data from both policies will go into the best response's replay buffer.
            # Here we ensure policies not from the best response have the exact same preprocessing as the best response.
            for average_policy_id, br_policy_id in [("average_policy_0", "best_response_0"), ("average_policy_1", "best_response_1")]:
                if policy_id == average_policy_id:

                    if "action_probs" in postprocessed_batch:
                        del postprocessed_batch.data["action_probs"]
                    if "behaviour_logits" in postprocessed_batch:
                        del postprocessed_batch.data["behaviour_logits"]

                    br_policy: Policy = policies[br_policy_id]

                    new_batch = br_policy.postprocess_trajectory(
                        sample_batch=postprocessed_batch,
                        other_agent_batches=original_batches,
                        episode=episode)
                    copy_attributes(src_obj=new_batch, dst_obj=postprocessed_batch)
                elif policy_id == br_policy_id:
                    if "q_values" in postprocessed_batch:
                        del postprocessed_batch.data["q_values"]
                    del postprocessed_batch.data["action_dist_inputs"]

                if policy_id in ("average_policy_0", "best_response_0"):
                    assert agent_id == 0
                if policy_id in ("average_policy_1", "best_response_1"):
                    assert agent_id == 1

        def on_sample_end(self, *, worker: "RolloutWorker", samples: SampleBatch, **kwargs):
            super().on_sample_end(worker=worker, samples=samples, **kwargs)
            assert isinstance(samples, MultiAgentBatch)

            for policy_samples in samples.policy_batches.values():
                del policy_samples.data["action_prob"]
                del policy_samples.data["action_logp"]

            for average_policy_id, br_policy_id in [("average_policy_0", "best_response_0"), ("average_policy_1", "best_response_1")]:
                for policy_id, policy_samples in samples.policy_batches.items():
                    if policy_id == br_policy_id:
                        store_to_avg_policy_buffer(MultiAgentBatch(policy_batches={
                            average_policy_id: policy_samples
                        }, env_steps=policy_samples.count))
                if average_policy_id in samples.policy_batches:

                    if br_policy_id in samples.policy_batches:
                        all_policies_samples = samples.policy_batches[br_policy_id].concat(
                            other=samples.policy_batches[average_policy_id])
                    else:
                        all_policies_samples = samples.policy_batches[average_policy_id]
                    del samples.policy_batches[average_policy_id]
                    samples.policy_batches[br_policy_id] = all_policies_samples


        def on_train_result(self, *, trainer, result: dict, **kwargs):
            super().on_train_result(trainer=trainer, result=result, **kwargs)
            # print(trainer.latest_avg_trainer_result.keys())
            # log(pretty_dict_str(trainer.latest_avg_trainer_result))
            # if trainer.latest_avg_trainer_result is not None:
            #     result["avg_trainer_info"] = trainer.latest_avg_trainer_result.get("info", {})
            training_iteration = result["training_iteration"]
            if (calculate_openspiel_metanash and
                    (training_iteration == 1 or training_iteration % calc_metanash_every_n_iters == 0)):
                openspiel_game_version = env_config["version"]
                local_avg_policy_0 = trainer.workers.local_worker().policy_map["average_policy_0"]
                local_avg_policy_1 = trainer.workers.local_worker().policy_map["average_policy_1"]
                exploitability = nfsp_measure_exploitability_nonlstm(
                    rllib_policies=[local_avg_policy_0, local_avg_policy_1],
                    poker_game_version=openspiel_game_version)
                result["z_avg_policy_exploitability"] = exploitability

                # check_local_avg_policy_0 = trainer.avg_trainer.workers.local_worker().policy_map["average_policy_0"]
                # check_local_avg_policy_1 = trainer.avg_trainer.workers.local_worker().policy_map["average_policy_1"]
                # check_exploitability = nfsp_measure_exploitability_nonlstm(
                #     rllib_policies=[check_local_avg_policy_0, check_local_avg_policy_1],
                #     poker_game_version="leduc_poker")
                # assert np.isclose(exploitability, check_exploitability), f"values are {exploitability} and {check_exploitability}"


    # def train_avg_policy(x: MultiAgentBatch, worker, *args, **kwargs):
    #     avg_train_results = worker.avg_trainer.train()
    #     # br_trainer.set_weights(avg_trainer.get_weights(["average_policy_0"]))
    #     # br_trainer.set_weights(avg_trainer.get_weights(["average_policy_1"]))
    #     br_trainer.latest_avg_trainer_result = copy.deepcopy(avg_train_results)
    #     return x

    br_trainer_config = {
        "log_level": "DEBUG",
        "callbacks": NFSPBestResponseCallbacks,
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
            "policies_to_train": ["best_response_0", "best_response_1"],
            "policies": {
                "average_policy_0": (policy_classes["average_policy"], tmp_env.observation_space, tmp_env.action_space, {
                    "model": avg_policy_model_config,
                    "explore": False,
                }),
                "best_response_0": (policy_classes["best_response"], tmp_env.observation_space, tmp_env.action_space, {}),

                "average_policy_1": (policy_classes["average_policy"], tmp_env.observation_space, tmp_env.action_space, {
                    "model": avg_policy_model_config,
                    "explore": False,
                }),
                "best_response_1": (policy_classes["best_response"], tmp_env.observation_space, tmp_env.action_space, {}),
            },
            "policy_mapping_fn": select_policy,
        },
    }
    br_trainer_config = merge_dicts(br_trainer_config, get_trainer_config(tmp_env.action_space))

    br_trainer = trainer_class(config=br_trainer_config,
                               logger_creator=get_trainer_logger_creator(base_dir=results_dir, scenario_name=scenario_name))

    # assert isinstance(br_trainer.workers.local_worker().policy_map["average_policy_1"].model, LeducDQNFullyConnectedNetwork)
    # assert isinstance(br_trainer.workers.local_worker().policy_map["average_policy_0"].model, LeducDQNFullyConnectedNetwork)
    # assert isinstance(avg_trainer.workers.local_worker().policy_map["average_policy_0"].model, LeducDQNFullyConnectedNetwork)
    # assert isinstance(avg_trainer.workers.local_worker().policy_map["average_policy_0"].model, LeducDQNFullyConnectedNetwork)

    br_trainer.latest_avg_trainer_result = None
    train_iter_count = 0

    for trainer in [br_trainer, avg_trainer]:
        for policy_id, policy in trainer.workers.local_worker().policy_map.items():
            policy.policy_id = policy_id

    br_trainer.workers.local_worker().policy_map["average_policy_0"] = avg_trainer.workers.local_worker().policy_map["average_policy_0"]
    br_trainer.workers.local_worker().policy_map["average_policy_1"] = avg_trainer.workers.local_worker().policy_map["average_policy_1"]
    # br_trainer.set_weights(avg_trainer.get_weights(["average_policy_0"]))
    # br_trainer.set_weights(avg_trainer.get_weights(["average_policy_1"]))

    stopping_condition: StoppingCondition = nfsp_get_stopping_condition()

    print("starting")
    while True:
        print("avg train...")
        avg_train_results = avg_trainer.train()
        # br_trainer.set_weights(avg_trainer.get_weights(["average_policy_0"]))
        # br_trainer.set_weights(avg_trainer.get_weights(["average_policy_1"]))
        br_trainer.latest_avg_trainer_result = copy.deepcopy(avg_train_results)
        print("br train...")
        train_iter_results = br_trainer.train()  # do a step (or several) in the main RL loop


        train_iter_count += 1
        print("printing results..")
        if print_train_results:
            # Delete verbose debugging info before printing
            if "hist_stats" in train_iter_results:
                del train_iter_results["hist_stats"]
            if "td_error" in train_iter_results["info"]["learner"]["best_response_0"]:
                del train_iter_results["info"]["learner"]["best_response_0"]["td_error"]
            if "td_error" in train_iter_results["info"]["learner"]["best_response_1"]:
                del train_iter_results["info"]["learner"]["best_response_1"]["td_error"]
            log(pretty_dict_str(train_iter_results))
        log(f"Trainer logdir is {br_trainer.logdir}")

        if stopping_condition.should_stop_this_iter(latest_trainer_result=train_iter_results):
            print("stopping condition met.")
            break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str)
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(grl.__file__), "data")
    print(f"results dir is {results_dir}")

    train_poker_off_policy_rl_nfsp(
        results_dir=results_dir,
        scenario_name=args.scenario,
        print_train_results=True,
    )