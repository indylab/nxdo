import os
import time
import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Callable, Type, Dict
import tempfile
import argparse
from copy import deepcopy

from gym.spaces import Space, Discrete
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
from grl.rl_apps.scenarios.stopping_conditions import StoppingCondition
from grl.p2sro.payoff_table import PayoffTableStrategySpec
from grl.xfdo.restricted_game import RestrictedGame
from grl.xfdo.action_space_conversion import RestrictedToBaseGameActionSpaceConverter
from grl.rl_apps.xfdo.poker_utils import xfdo_nfsp_measure_exploitability_nonlstm
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


def save_nfsp_avg_policy_checkpoint(trainer: Trainer,
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
    br_weights = {k.replace(".", "_dot_"): v for k, v in br_weights.items()} # periods cause HDF5 NaturalNaming warnings
    ensure_dir(file_path=checkpoint_path)
    deepdish.io.save(path=checkpoint_path, data={
        "weights": br_weights,
        "date_time_str": date_time,
        "seconds_since_epoch": time.time(),
        "timesteps_training": timesteps_training,
        "episodes_training": episodes_training
    }, )
    return checkpoint_path

def create_get_pure_strat_cached(cache: dict):
    def load_pure_strat_cached(policy: Policy, pure_strat_spec):

        pure_strat_checkpoint_path = pure_strat_spec.metadata["checkpoint_path"]

        if pure_strat_checkpoint_path in cache:
            weights = cache[pure_strat_checkpoint_path]
        else:
            checkpoint_data = deepdish.io.load(path=pure_strat_checkpoint_path)
            weights = checkpoint_data["weights"]
            weights = {k.replace("_dot_", "."): v for k, v in weights.items()}

            weights = convert_to_torch_tensor(weights, device=policy.device)
            cache[pure_strat_checkpoint_path] = weights
        # policy.set_weights(weights=weights)
        policy.model.load_state_dict(weights)

        policy.policy_spec = pure_strat_spec
    return load_pure_strat_cached

def train_off_policy_rl_nfsp_restricted_game(results_dir: str,
                             scenario: dict,
                             player_to_base_game_action_specs: Dict[int, List[PayoffTableStrategySpec]],
                             stopping_condition: StoppingCondition,
                             print_train_results: bool = True):

    env_class = scenario["env_class"]
    base_env_config = scenario["env_config"]
    trainer_class = scenario["trainer_class_nfsp"]
    avg_trainer_class = scenario["avg_trainer_class_nfsp"]
    policy_classes: Dict[str, Type[Policy]] = scenario["policy_classes_nfsp"]
    anticipatory_param: float = scenario["anticipatory_param_nfsp"]
    get_trainer_config = scenario["get_trainer_config_nfsp"]
    get_avg_trainer_config = scenario["get_avg_trainer_config_nfsp"]
    calculate_openspiel_metanash: bool = scenario["calculate_openspiel_metanash"]
    calc_metanash_every_n_iters: int = scenario["calc_metanash_every_n_iters"]
    metrics_smoothing_episodes_override: int = scenario["metanash_metrics_smoothing_episodes_override"]

    assert scenario["xfdo_metanash_method"] == "nfsp"

    ray.init(log_to_driver=os.getenv("RAY_LOG_TO_DRIVER", False), address='auto', _redis_password='5241590000000000', ignore_reinit_error=True, local_mode=False)

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

    def _create_base_env():
        return env_class(env_config=base_env_config)

    tmp_base_env = _create_base_env()

    restricted_env_config = {"create_env_fn": _create_base_env}
    tmp_env = RestrictedGame(env_config=restricted_env_config)
    restricted_game_action_spaces = [Discrete(n=len(player_to_base_game_action_specs[p])) for p in range(2)]

    assert all(restricted_game_action_spaces[0] == space for space in restricted_game_action_spaces)

    avg_trainer_config = merge_dicts({
        "log_level": "DEBUG",
        "framework": "torch",
        "env": RestrictedGame,
        "env_config": restricted_env_config,
        "num_gpus": 0.0,
        "num_gpus_per_worker": 0.0,
        "num_workers": 0,
        "num_envs_per_worker": 1,
        "multiagent": {
            "policies_to_train": ["average_policy_0", "average_policy_1"],
            "policies": {
                "average_policy_0": (policy_classes["average_policy"], tmp_env.observation_space, restricted_game_action_spaces[0], {"explore": False}),
                "average_policy_1": (policy_classes["average_policy"], tmp_env.observation_space, restricted_game_action_spaces[1], {"explore": False}),
                "delegate_policy": (policy_classes["delegate_policy"], tmp_base_env.observation_space, tmp_env.base_action_space, {"explore": False}),
            },
            "policy_mapping_fn": assert_not_called,
        },

    }, get_avg_trainer_config(restricted_game_action_spaces[0]))

    avg_trainer = avg_trainer_class(config=avg_trainer_config, logger_creator=get_trainer_logger_creator(base_dir=results_dir,
                                                 scenario_name=f"nfsp_restricted_game_avg_trainer"))

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
                    if "action_probs" in postprocessed_batch:
                        del postprocessed_batch.data["action_probs"]
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
            training_iteration = result["training_iteration"]

            if "policy_reward_mean" in result:
                br_0_rew = result["policy_reward_mean"]["best_response_0"]
                br_1_rew = result["policy_reward_mean"]["best_response_1"]
                avg_br_reward = (br_0_rew + br_1_rew) / 2.0
                result["avg_br_reward_both_players"] = avg_br_reward
            else:
                assert training_iteration < 5

            # print(trainer.latest_avg_trainer_result.keys())
            # log(pretty_dict_str(trainer.latest_avg_trainer_result))
            # if trainer.latest_avg_trainer_result is not None:
            #     result["avg_trainer_info"] = trainer.latest_avg_trainer_result.get("info", {})
            if (calculate_openspiel_metanash and
                    (training_iteration == 1 or training_iteration % calc_metanash_every_n_iters == 0)):
                openspiel_game_version = base_env_config["version"]
                local_avg_policy_0 = trainer.workers.local_worker().policy_map["average_policy_0"]
                local_avg_policy_1 = trainer.workers.local_worker().policy_map["average_policy_1"]
                exploitability = xfdo_nfsp_measure_exploitability_nonlstm(
                    rllib_policies=[local_avg_policy_0, local_avg_policy_1],
                    poker_game_version=openspiel_game_version,
                    action_space_converters=trainer.get_local_converters()
                )
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
        "env": RestrictedGame,
        "env_config": restricted_env_config,
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
                "average_policy_0": (policy_classes["average_policy"], tmp_env.observation_space, restricted_game_action_spaces[0], {"explore": False}),
                "best_response_0": (policy_classes["best_response"], tmp_env.observation_space, restricted_game_action_spaces[0], {}),

                "average_policy_1": (policy_classes["average_policy"], tmp_env.observation_space, restricted_game_action_spaces[1], {"explore": False}),
                "best_response_1": (policy_classes["best_response"], tmp_env.observation_space, restricted_game_action_spaces[1], {}),

                "delegate_policy": (policy_classes["delegate_policy"], tmp_base_env.observation_space, tmp_env.base_action_space, {"explore": False}),

            },
            "policy_mapping_fn": select_policy,
        },
    }

    assert all(restricted_game_action_spaces[0] == space for space in restricted_game_action_spaces), \
        "If not true, the line below with \"get_trainer_config\" may need to be changed to a better solution."
    br_trainer_config = merge_dicts(br_trainer_config, get_trainer_config(restricted_game_action_spaces[0]))

    br_trainer_config["metrics_smoothing_episodes"] = metrics_smoothing_episodes_override

    br_trainer = trainer_class(config=br_trainer_config,
                               logger_creator=get_trainer_logger_creator(base_dir=results_dir,
                                                                         scenario_name="nfsp_restricted_game_trainer"))

    weights_cache = {}
    for _trainer in [br_trainer, avg_trainer]:
        def _set_worker_converters(worker: RolloutWorker):
            worker_delegate_policy = worker.policy_map["delegate_policy"]
            player_converters = []
            for player in range(2):
                player_converter = RestrictedToBaseGameActionSpaceConverter(
                    delegate_policy=worker_delegate_policy, policy_specs=player_to_base_game_action_specs[player],
                    load_policy_spec_fn=create_get_pure_strat_cached(cache=weights_cache))
                player_converters.append(player_converter)
                worker.foreach_env(lambda env: env.set_action_conversion(player, player_converter))
            worker_delegate_policy.player_converters = player_converters

        _trainer.workers.foreach_worker(_set_worker_converters)
        _trainer.get_local_converters = lambda: trainer.workers.local_worker().policy_map[
            "delegate_policy"].player_converters

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

    if restricted_game_action_spaces[0].n == 1:
        final_train_result = {"episodes_total": 0, "timesteps_total": 0, "training_iteration": 0}
        tmp_callback = NFSPBestResponseCallbacks()
        tmp_callback.on_train_result(trainer=br_trainer, result=final_train_result)
        print(f"\n\nexploitability: {final_train_result['z_avg_policy_exploitability']}\n\n")
    else:
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
                final_train_result = deepcopy(train_iter_results)
                break

    avg_policy_specs = []
    for player in range(2):
        strategy_id = f"avg_policy_player_{player}_{datetime_str()}"

        checkpoint_path = save_nfsp_avg_policy_checkpoint(trainer=br_trainer, policy_id_to_save=f"average_policy_{player}",
                                        save_dir=checkpoint_dir(trainer=br_trainer),
                                        timesteps_training=final_train_result["timesteps_total"],
                                        episodes_training=final_train_result["episodes_total"],
                                        checkpoint_name=f"{strategy_id}.h5")

        avg_policy_spec = PayoffTableStrategySpec(
            strategy_id=strategy_id,
            metadata={"checkpoint_path": checkpoint_path})
        avg_policy_specs.append(avg_policy_spec)

    avg_trainer.cleanup()
    br_trainer.cleanup()
    ray.shutdown()

    assert final_train_result is not None
    return avg_policy_specs, final_train_result
