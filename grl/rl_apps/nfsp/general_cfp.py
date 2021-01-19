import argparse
import logging
import os
import random
import tempfile
import time
from copy import deepcopy
from typing import List, Tuple, Type, Dict

import deepdish
import numpy as np
import ray
from ray.rllib.utils import merge_dicts, try_import_torch

torch, _ = try_import_torch()

from ray.rllib.agents import Trainer
from ray.rllib.agents.sac import SACTrainer, SACTorchPolicy
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.utils.typing import TensorType, PolicyID
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper

import grl
from grl.utils import pretty_dict_str, datetime_str, ensure_dir

from grl.p2sro.payoff_table import PayoffTableStrategySpec
from grl.rl_apps.nfsp.openspiel_utils import snfsp_measure_exploitability_nonlstm

from grl.rllib_tools.space_saving_logger import SpaceSavingLogger
from grl.rl_apps.scenarios.poker import scenarios
from grl.rl_apps.scenarios.stopping_conditions import StoppingCondition
from grl.nfsp_rllib.checkpoint_reservoir_buffer import ReservoirBuffer

logger = logging.getLogger(__name__)




def save_cfp_best_response_checkpoint(trainer: Trainer,
                                      policy_id_to_save: str,
                                      save_dir: str,
                                      timesteps_training_br: int,
                                      episodes_training_br: int,
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
        "timesteps_training_br": timesteps_training_br,
        "episodes_training_br": episodes_training_br
    }, )
    return checkpoint_path


def load_pure_strat(policy: Policy, pure_strat_spec, checkpoint_path: str = None):
    assert pure_strat_spec is None or checkpoint_path is None, "can only pass one or the other"
    if checkpoint_path is None:
        pure_strat_checkpoint_path = pure_strat_spec.metadata["checkpoint_path"]
    else:
        pure_strat_checkpoint_path = checkpoint_path
    checkpoint_data = deepdish.io.load(path=pure_strat_checkpoint_path)
    weights = checkpoint_data["weights"]
    weights = {k.replace("_dot_", "."): v for k, v in weights.items()}
    policy.set_weights(weights=weights)
    policy.policy_spec = pure_strat_spec


def create_metadata_with_new_checkpoint_for_current_best_response(br_trainer: Trainer,
                                                                policy_id_to_save: str,
                                                              save_dir: str,
                                                              timesteps_training_br: int,
                                                              episodes_training_br: int,
                                                              checkpoint_name=None
                                                              ):
    return {
        "checkpoint_path": save_cfp_best_response_checkpoint(trainer=br_trainer,
                                                             policy_id_to_save=policy_id_to_save,
                                                             save_dir=save_dir,
                                                             timesteps_training_br=timesteps_training_br,
                                                             episodes_training_br=episodes_training_br,
                                                             checkpoint_name=checkpoint_name),
        "timesteps_training_br": timesteps_training_br,
        "episodes_training_br": episodes_training_br
    }

def add_best_response_strat_to_average_policy(recepient_trainer: Trainer, br_spec: PayoffTableStrategySpec, reservoir_buffer_idx: int, avg_policy_id):
    def worker_add_spec(worker: RolloutWorker):
        avg_policy: Policy = worker.policy_map[avg_policy_id]
        if not hasattr(avg_policy, "br_specs"):
            avg_policy.br_specs = ReservoirBuffer()
        avg_policy.br_specs.add(br_spec, idx=reservoir_buffer_idx)
    recepient_trainer.workers.foreach_worker(worker_add_spec)


def get_trainer_logger_creator(base_dir: str, scenario_name: str):
    logdir_prefix = f"{scenario_name.__name__}_sparse_{datetime_str()}"

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


def train_cfp(results_dir: str, scenario_name: str, print_train_results: bool = True):
    try:
        scenario = scenarios[scenario_name]
    except KeyError:
        raise NotImplementedError(f"Unknown scenario name: \'{scenario_name}\'. Existing scenarios are:\n"
                                  f"{list(scenarios.keys())}")
    env_class = scenario["env_class"]
    env_config = scenario["env_config"]
    trainer_class = scenario["trainer_class"]
    policy_classes: Dict[str, Type[Policy]] = scenario["policy_classes"]
    anticipatory_param: float = scenario["anticipatory_param"]
    checkpoint_reservoir_size: int = scenario["checkpoint_reservoir_size"]
    get_trainer_config = scenario["get_trainer_config"]
    calculate_openspiel_metanash: bool = scenario["calculate_openspiel_metanash"]
    calc_metanash_every_n_iters: int = scenario["calc_metanash_every_n_iters"]
    cfp_get_stopping_condition = scenario["cfp_get_stopping_condition"]

    ray.init(ignore_reinit_error=True, local_mode=False)

    def log(message, level=logging.INFO):
        logger.log(level, message)

    def get_select_policy(trainer_player_num: int):
        def select_policy(agent_id: int):
            if agent_id == trainer_player_num:
                return "best_response"
            elif agent_id == 1 - trainer_player_num:
                return "opponent_average_policy"
            else:
                raise ValueError(f"unexpected agent_id: {agent_id}")
        return select_policy

    tmp_env = env_class(env_config=env_config)

    class ParallelSNFSPBestResponseCallbacks(DefaultCallbacks):

        def on_train_result(self, *, trainer, result: dict, **kwargs):
            super().on_train_result(trainer=trainer, result=result, **kwargs)
            result["scenario_name"] = trainer.scenario_name

            # print(trainer.latest_avg_trainer_result.keys())
            # log(pretty_dict_str(trainer.latest_avg_trainer_result))
            # if trainer.latest_avg_trainer_result is not None:
            #     result["avg_trainer_info"] = trainer.latest_avg_trainer_result.get("info", {})

            if "hist_stats" in result:
                del result["hist_stats"]
            if "td_error" in result["info"]["learner"]["best_response"]:
                del result["info"]["learner"]["best_response"]["td_error"]

            result["br_player"] = trainer.player

            if trainer.player == 0:
                result["other_trainer_results"] = trainer.other_player_latest_train_result
                if len(trainer.other_player_latest_train_result) > 0:
                    result["timesteps_total"] += trainer.other_player_latest_train_result["timesteps_total"]
                    result["episodes_total"] += trainer.other_player_latest_train_result["episodes_total"]

                br_trainer_0 = trainer
                br_trainer_1 = trainer.other_trainer

                training_iteration = result["training_iteration"]
                if (calculate_openspiel_metanash and
                        (training_iteration == 1 or training_iteration % calc_metanash_every_n_iters == 0)):

                    local_avg_policy_0 = br_trainer_1.workers.local_worker().policy_map["opponent_average_policy"]
                    local_avg_policy_1 = br_trainer_0.workers.local_worker().policy_map["opponent_average_policy"]

                    br_checkpoint_path_tuple_list: List[Tuple[str, str]] = []
                    for br_spec_0, br_spec_1 in zip(local_avg_policy_0.br_specs, local_avg_policy_1.br_specs):
                        br_checkpoint_path_tuple_list.append((
                            br_spec_0.metadata["checkpoint_path"],
                            br_spec_1.metadata["checkpoint_path"]
                        ))

                    exploitability = snfsp_measure_exploitability_nonlstm(
                        br_checkpoint_path_tuple_list=br_checkpoint_path_tuple_list,
                        set_policy_weights_fn=lambda policy, path: load_pure_strat(policy=policy, checkpoint_path=path, pure_strat_spec=None),
                        rllib_policies=[local_avg_policy_0, local_avg_policy_1],
                        poker_game_version="kuhn_poker")
                    result["z_avg_policy_exploitability"] = exploitability

        def on_episode_start(self, *, worker: "RolloutWorker", base_env: BaseEnv, policies: Dict[PolicyID, Policy],
                             episode: MultiAgentEpisode, env_index: int, **kwargs):
            super().on_episode_start(worker=worker, base_env=base_env, policies=policies, episode=episode,
                                     env_index=env_index, **kwargs)
            avg_policy: Policy = worker.policy_map["opponent_average_policy"]
            if not hasattr(avg_policy, "br_specs"):
                return
            new_br_spec = avg_policy.br_specs.sample()[0]
            if hasattr(avg_policy, "policy_spec") and new_br_spec == avg_policy.policy_spec:
                return
            load_pure_strat(policy=avg_policy, pure_strat_spec=new_br_spec)

    br_trainer_configs = [merge_dicts({
        "log_level": "DEBUG",
        "callbacks": ParallelSNFSPBestResponseCallbacks,
        "env": env_class,
        "env_config": env_config,
        "gamma": 1.0,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        # "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_gpus": 0,
        "num_workers": 0,
        "num_gpus_per_worker": 0,
        "num_envs_per_worker": 1,
        "multiagent": {
            "policies_to_train": ["best_response"],
            "policies": {
                "best_response": (policy_classes["best_response"], tmp_env.observation_space, tmp_env.action_space, {}),
                "opponent_average_policy": (policy_classes["best_response"], tmp_env.observation_space, tmp_env.action_space, {"explore": False}),
            },
            "policy_mapping_fn": get_select_policy(trainer_player_num=player),
        },
    }, get_trainer_config(tmp_env.action_space)) for player in range(2)]

    br_trainers: List[Trainer] = [trainer_class(config=br_trainer_configs[player],
                                    logger_creator=get_trainer_logger_creator(base_dir=results_dir,
                                                                              scenario_name=f"{scenario_name}_trainer_{player}"))
                    for player in range(2)]

    for player, br_trainer in enumerate(br_trainers):
        other_player = 1-player
        br_trainer.player = player
        br_trainer.other_trainer = br_trainers[other_player]
        br_trainer.other_player_latest_train_result = {}

        # scenario_name logged in on_train_result_callback
        br_trainer.scenario_name = scenario_name

    def checkpoint_brs_for_average_policy(checkpoint_count: int, both_train_iter_results=(None, None)):
        for player in range(2):
            br_trainer = br_trainers[player]
            br_train_iter_results = both_train_iter_results[player]
            other_player = 1 - player
            other_trainer = br_trainers[other_player]

            log("\nCHECKPOINTING BR\n")
            if br_train_iter_results is not None:
                total_timesteps_training_br = br_train_iter_results["timesteps_total"]
                total_episodes_training_br = br_train_iter_results["episodes_total"]
            else:
                total_timesteps_training_br = 0
                total_episodes_training_br = 0

            local_avg_policy = other_trainer.workers.local_worker().policy_map["opponent_average_policy"]
            if not hasattr(local_avg_policy, "br_specs"):
                local_avg_policy.br_specs = ReservoirBuffer(reservoir_buffer_capacity=checkpoint_reservoir_size)
            can_add, add_idx = local_avg_policy.br_specs.ask_to_add()
            if can_add:
                old_spec = None
                if add_idx is not None:
                    old_spec = local_avg_policy.br_specs[add_idx]

                checkpoint_metadata = create_metadata_with_new_checkpoint_for_current_best_response(
                    policy_id_to_save="best_response",
                    br_trainer=br_trainer,
                    save_dir=checkpoint_dir(trainer=br_trainer),
                    timesteps_training_br=total_timesteps_training_br,
                    episodes_training_br=total_episodes_training_br,
                    checkpoint_name=f"best_response_{checkpoint_count}_{datetime_str()}.h5"
                )
                log(checkpoint_metadata)

                br_checkpoint_spec = PayoffTableStrategySpec(strategy_id=str(checkpoint_count),
                                                             metadata=checkpoint_metadata)
                add_best_response_strat_to_average_policy(recepient_trainer=other_trainer, br_spec=br_checkpoint_spec,
                                                          reservoir_buffer_idx=add_idx,
                                                          avg_policy_id="opponent_average_policy")

                if old_spec is not None:
                    # delete the checkpoint that we're replacing
                    checkpoint_path = old_spec.metadata["checkpoint_path"]
                    os.remove(checkpoint_path)

    train_iter_count = 0
    checkpoint_count = 0
    checkpoint_every_n_iters = 1
    print("starting")
    checkpoint_brs_for_average_policy(checkpoint_count=checkpoint_count, both_train_iter_results=(None, None))

    # train_thread_executor = ThreadPoolExecutor(max_workers=2)

    stopping_condition: StoppingCondition = cfp_get_stopping_condition()

    br_trainer_0, br_trainer_1 = br_trainers
    assert br_trainer_0.player == 0
    assert br_trainer_1.player == 1

    while True:

        # do a train step for both BRs at the same time
        # train_result_futures = tuple(train_thread_executor.submit(lambda: trainer.train()) for trainer in br_trainers)
        # both_train_iter_results = tuple(future.result() for future in train_result_futures)

        both_train_iter_results = tuple(trainer.train() for trainer in br_trainers)

        if train_iter_count % checkpoint_every_n_iters == 0:
            checkpoint_brs_for_average_policy(checkpoint_count=checkpoint_count, both_train_iter_results=both_train_iter_results)
            checkpoint_count += 1

        train_iter_count += 1
        print("printing results..")

        for br_trainer, train_iter_results in zip(br_trainers, both_train_iter_results):
            print(f"trainer {br_trainer.player}")
            # Delete verbose debugging info
            if "hist_stats" in train_iter_results:
                del train_iter_results["hist_stats"]
            if "td_error" in train_iter_results["info"]["learner"]["best_response"]:
                del train_iter_results["info"]["learner"]["best_response"]["td_error"]
            assert br_trainer.player in [0,1]
            log(f"Trainer {br_trainer.player} logdir is {br_trainer.logdir}")

        assert br_trainer_1.other_trainer.player == 0, br_trainer_1.other_trainer.player
        br_trainer_1.other_trainer.other_player_latest_train_result = deepcopy(both_train_iter_results[1])
        assert len(both_train_iter_results[1]) > 0

        if print_train_results:
            log(pretty_dict_str(both_train_iter_results[0]))

        if stopping_condition.should_stop_this_iter(latest_trainer_result=both_train_iter_results[0]):
            print("stopping condition met.")
            break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str)
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(grl.__file__), "data")
    print(f"results dir is {results_dir}")

    train_cfp(
        results_dir=results_dir,
        scenario_name=args.scenario,
        print_train_results=True,
    )