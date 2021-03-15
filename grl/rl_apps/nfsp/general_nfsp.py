import argparse
import copy
import logging
import os
import tempfile
import time
from typing import Any, Tuple, Type, Dict, Union

from termcolor import colored
import deepdish
import numpy as np
import ray
from ray.rllib.utils import merge_dicts, try_import_torch

torch, _ = try_import_torch()

from ray.rllib.agents import Trainer
from ray.rllib.utils.typing import AgentID, PolicyID
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
import grl
from grl.utils.common import pretty_dict_str, datetime_str, ensure_dir, copy_attributes

from grl.algos.nfsp_rllib.nfsp import get_store_to_avg_policy_buffer_fn
from grl.rl_apps.nfsp.openspiel_utils import nfsp_measure_exploitability_nonlstm
from grl.rllib_tools.space_saving_logger import SpaceSavingLogger
from grl.rl_apps.scenarios.catalog import scenario_catalog
from grl.rl_apps.scenarios.nfsp_scenario import NFSPScenario
from grl.rl_apps.scenarios.ray_setup import init_ray_for_scenario
from grl.rl_apps.scenarios.stopping_conditions import StoppingCondition
from grl.utils.strategy_spec import StrategySpec

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
    return os.path.join(trainer.logdir, "avg_policy_checkpoints")


def spec_checkpoint_dir(trainer: Trainer):
    return os.path.join(trainer.logdir, "avg_policy_checkpoint_specs")


def save_nfsp_average_policy_checkpoint(trainer: Trainer,
                                        policy_id_to_save: str,
                                        save_dir: str,
                                        timesteps_training: int,
                                        episodes_training: int,
                                        checkpoint_name=None):
    policy_name = policy_id_to_save
    date_time = datetime_str()
    if checkpoint_name is None:
        checkpoint_name = f"policy_{policy_name}_{date_time}.h5"
    checkpoint_path = os.path.join(save_dir, checkpoint_name)
    br_weights = trainer.get_weights([policy_id_to_save])[policy_id_to_save]
    br_weights = {k.replace(".", "_dot_"): v for k, v in
                  br_weights.items()}  # periods cause HDF5 NaturalNaming warnings
    ensure_dir(file_path=checkpoint_path)
    deepdish.io.save(path=checkpoint_path, data={
        "weights": br_weights,
        "date_time_str": date_time,
        "seconds_since_epoch": time.time(),
        "timesteps_training": timesteps_training,
        "episodes_training": episodes_training
    }, )
    return checkpoint_path


def create_metadata_with_new_checkpoint(br_trainer: Trainer,
                                        policy_id_to_save: str,
                                        save_dir: str,
                                        timesteps_training: int,
                                        episodes_training: int,
                                        checkpoint_name=None
                                        ):
    return {
        "checkpoint_path": save_nfsp_average_policy_checkpoint(trainer=br_trainer,
                                                               policy_id_to_save=policy_id_to_save,
                                                               save_dir=save_dir,
                                                               timesteps_training=timesteps_training,
                                                               episodes_training=episodes_training,
                                                               checkpoint_name=checkpoint_name),
        "timesteps_training": timesteps_training,
        "episodes_training": episodes_training
    }


@ray.remote(num_cpus=0)
class StatDeque(object):
    def __init__(self, max_items: int):
        self._data = []
        self._max_items = max_items

    def add(self, item):
        self._data.append(item)
        if len(self._data) > self._max_items:
            del self._data[0]

    def get_mean(self):
        return np.mean(self._data)


def train_off_policy_rl_nfsp(results_dir: str,
                             scenario_name: str,
                             print_train_results: bool = True):
    
    scenario: NFSPScenario = scenario_catalog.get(scenario_name=scenario_name)

    env_class = scenario.env_class
    env_config = scenario.env_config
    trainer_class = scenario.trainer_class
    avg_trainer_class = scenario.avg_trainer_class
    policy_classes: Dict[str, Type[Policy]] = scenario.policy_classes
    anticipatory_param: float = scenario.anticipatory_param
    get_trainer_config = scenario.get_trainer_config
    get_avg_trainer_config = scenario.get_avg_trainer_config
    calculate_openspiel_metanash: bool = scenario.calculate_openspiel_metanash
    calc_metanash_every_n_iters: int = scenario.calc_metanash_every_n_iters
    checkpoint_every_n_iters: Union[int, None] = scenario.checkpoint_every_n_iters
    nfsp_get_stopping_condition = scenario.nfsp_get_stopping_condition

    init_ray_for_scenario(scenario=scenario, head_address=None, logging_level=logging.INFO)

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
    open_spiel_env_config = tmp_env.open_spiel_env_config if calculate_openspiel_metanash else None

    avg_policy_model_config = get_trainer_config(tmp_env)["model"]

    avg_trainer_config = merge_dicts({
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
                "average_policy_0": (
                policy_classes["average_policy"], tmp_env.observation_space, tmp_env.action_space, {
                    "model": avg_policy_model_config
                }),
                "average_policy_1": (
                policy_classes["average_policy"], tmp_env.observation_space, tmp_env.action_space, {
                    "model": avg_policy_model_config
                }),
            },
            "policy_mapping_fn": assert_not_called,
        },

    }, get_avg_trainer_config(tmp_env))

    avg_trainer = avg_trainer_class(config=avg_trainer_config,
                                    logger_creator=get_trainer_logger_creator(base_dir=results_dir,
                                                                              scenario_name=f"{scenario_name}_avg_trainer"))

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
            for average_policy_id, br_policy_id in [("average_policy_0", "best_response_0"),
                                                    ("average_policy_1", "best_response_1")]:
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
                    if "action_probs" in postprocessed_batch:
                        del postprocessed_batch.data["action_probs"]
                    del postprocessed_batch.data["action_dist_inputs"]

                if policy_id in ("average_policy_0", "best_response_0"):
                    assert agent_id == 0
                if policy_id in ("average_policy_1", "best_response_1"):
                    assert agent_id == 1

        def on_episode_end(self, *, worker: "RolloutWorker", base_env: BaseEnv, policies: Dict[PolicyID, Policy],
                           episode: MultiAgentEpisode, env_index: int, **kwargs):
            super().on_episode_end(worker=worker, base_env=base_env, policies=policies, episode=episode,
                                   env_index=env_index, **kwargs)

            episode_policies = set(episode.agent_rewards.keys())
            if episode_policies == {(0, "average_policy_0"), (1, "best_response_1")}:
                worker.avg_br_reward_deque.add.remote(episode.agent_rewards[(1, "best_response_1")])
            elif episode_policies == {(1, "average_policy_1"), (0, "best_response_0")}:
                worker.avg_br_reward_deque.add.remote(episode.agent_rewards[(0, "best_response_0")])

        def on_sample_end(self, *, worker: "RolloutWorker", samples: SampleBatch, **kwargs):
            super().on_sample_end(worker=worker, samples=samples, **kwargs)
            assert isinstance(samples, MultiAgentBatch)

            for policy_samples in samples.policy_batches.values():
                if "action_prob" in policy_samples.data:
                    del policy_samples.data["action_prob"]
                if "action_logp" in policy_samples.data:
                    del policy_samples.data["action_logp"]

            for average_policy_id, br_policy_id in [("average_policy_0", "best_response_0"),
                                                    ("average_policy_1", "best_response_1")]:
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
            result["scenario_name"] = trainer.scenario_name

            result["avg_br_reward_both_players"] = ray.get(trainer.avg_br_reward_deque.get_mean.remote())

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
                    poker_game_version=openspiel_game_version,
                    open_spiel_env_config=open_spiel_env_config
                )
                result["z_avg_policy_exploitability"] = exploitability
                logger.info(colored(
                    f"(Graph this in a notebook) Exploitability: {exploitability} - Saving exploitability stats "
                    f"to {os.path.join(trainer.logdir, 'result.json')}", "green"))

            if checkpoint_every_n_iters and training_iteration % checkpoint_every_n_iters == 0:
                for player in range(2):
                    checkpoint_metadata = create_metadata_with_new_checkpoint(
                        policy_id_to_save=f"average_policy_{player}",
                        br_trainer=br_trainer,
                        save_dir=checkpoint_dir(trainer=br_trainer),
                        timesteps_training=result["timesteps_total"],
                        episodes_training=result["episodes_total"],
                        checkpoint_name=f"average_policy_player_{player}_iter_{training_iteration}.h5"
                    )
                    avg_pol_checkpoint_spec = StrategySpec(
                        strategy_id=f"avg_pol_player_{player}_iter_{training_iteration}",
                        metadata=checkpoint_metadata)
                    checkpoint_path = os.path.join(spec_checkpoint_dir(br_trainer),
                                                   f"average_policy_player_{player}_iter_{training_iteration}.json")
                    ensure_dir(checkpoint_path)
                    with open(checkpoint_path, "+w") as checkpoint_spec_file:
                        checkpoint_spec_file.write(avg_pol_checkpoint_spec.to_json())

    br_trainer_config = {
        "log_level": "DEBUG",
        "callbacks": NFSPBestResponseCallbacks,
        "env": env_class,
        "env_config": env_config,
        "gamma": 1.0,
        "num_gpus": 0.0,
        "num_workers": 0,
        "num_gpus_per_worker": 0.0,
        "num_envs_per_worker": 1,
        "multiagent": {
            "policies_to_train": ["best_response_0", "best_response_1"],
            "policies": {
                "average_policy_0": (
                policy_classes["average_policy"], tmp_env.observation_space, tmp_env.action_space, {
                    "model": avg_policy_model_config,
                    "explore": False,
                }),
                "best_response_0": (
                policy_classes["best_response"], tmp_env.observation_space, tmp_env.action_space, {}),

                "average_policy_1": (
                policy_classes["average_policy"], tmp_env.observation_space, tmp_env.action_space, {
                    "model": avg_policy_model_config,
                    "explore": False,
                }),
                "best_response_1": (
                policy_classes["best_response"], tmp_env.observation_space, tmp_env.action_space, {}),
            },
            "policy_mapping_fn": select_policy,
        },
    }
    br_trainer_config = merge_dicts(br_trainer_config, get_trainer_config(tmp_env))

    br_trainer = trainer_class(config=br_trainer_config,
                               logger_creator=get_trainer_logger_creator(base_dir=results_dir,
                                                                         scenario_name=scenario_name))

    avg_br_reward_deque = StatDeque.remote(max_items=br_trainer_config["metrics_smoothing_episodes"])

    def _set_avg_br_rew_deque(worker: RolloutWorker):
        worker.avg_br_reward_deque = avg_br_reward_deque

    br_trainer.workers.foreach_worker(_set_avg_br_rew_deque)
    br_trainer.avg_br_reward_deque = avg_br_reward_deque

    # scenario_name logged in on_train_result_callback
    br_trainer.scenario_name = scenario_name

    br_trainer.latest_avg_trainer_result = None
    train_iter_count = 0

    for trainer in [br_trainer, avg_trainer]:
        for policy_id, policy in trainer.workers.local_worker().policy_map.items():
            policy.policy_id = policy_id

    avg_weights = avg_trainer.get_weights(["average_policy_0", "average_policy_1"])
    br_trainer.workers.foreach_worker(lambda worker: worker.set_weights(avg_weights))

    stopping_condition: StoppingCondition = nfsp_get_stopping_condition()

    print("starting")
    while True:
        print("avg train...")
        avg_train_results = avg_trainer.train()
        avg_weights = avg_trainer.get_weights(["average_policy_0", "average_policy_1"])
        br_trainer.workers.foreach_worker(lambda worker: worker.set_weights(avg_weights))
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

    train_off_policy_rl_nfsp(
        results_dir=results_dir,
        scenario_name=args.scenario,
        print_train_results=True,
    )
