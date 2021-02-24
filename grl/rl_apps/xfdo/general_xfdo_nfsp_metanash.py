import ray
from ray.rllib.utils import merge_dicts, try_import_torch

torch, _ = try_import_torch()

import os
import time
import logging
import numpy as np
from typing import List, Any, Tuple, Type, Dict
from copy import deepcopy

from gym.spaces import Discrete
import copy
import deepdish

from ray.rllib.agents import Trainer

from ray.rllib.utils.typing import AgentID, PolicyID
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
from grl.rl_apps.scenarios.nxdo_scenario import NXDOScenario
from grl.utils.common import pretty_dict_str, datetime_str, ensure_dir, copy_attributes
from grl.rllib_tools.stat_deque import StatDeque
from grl.nfsp_rllib.nfsp import get_store_to_avg_policy_buffer_fn
from grl.rllib_tools.space_saving_logger import SpaceSavingLogger, get_trainer_logger_creator
from grl.rl_apps.scenarios.stopping_conditions import StoppingCondition
from grl.utils.strategy_spec import StrategySpec
from grl.xfdo.restricted_game import RestrictedGame
from grl.xfdo.action_space_conversion import RestrictedToBaseGameActionSpaceConverter
from grl.rl_apps.xfdo.poker_utils import xfdo_nfsp_measure_exploitability_nonlstm
from grl.xfdo.opnsl_restricted_game import OpenSpielRestrictedGame, get_restricted_game_obs_conversions
from grl.rllib_tools.policy_checkpoints import save_policy_checkpoint, create_get_pure_strat_cached

from grl.rl_apps.scenarios.ray_setup import init_ray_for_scenario

logger = logging.getLogger(__name__)


def should_log_trainer_result(result: dict) -> bool:
    return ("z_avg_policy_exploitability" in result or
            result["training_iteration"] % 1000 == 0)

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


def train_off_policy_rl_nfsp_restricted_game(results_dir: str,
                                             scenario: NXDOScenario,
                                             player_to_base_game_action_specs: Dict[int, List[StrategySpec]],
                                             stopping_condition: StoppingCondition,
                                             manager_metadata: dict,
                                             print_train_results: bool = True):

    use_openspiel_restricted_game: bool = scenario.use_openspiel_restricted_game
    get_restricted_game_custom_model = scenario.get_restricted_game_custom_model
    env_class = scenario.env_class
    base_env_config = scenario.env_config
    trainer_class = scenario.trainer_class_nfsp
    avg_trainer_class = scenario.avg_trainer_class_nfsp
    policy_classes: Dict[str, Type[Policy]] = scenario.policy_classes_nfsp
    anticipatory_param: float = scenario.anticipatory_param_nfsp
    get_trainer_config = scenario.get_trainer_config_nfsp
    get_avg_trainer_config = scenario.get_avg_trainer_config_nfsp
    calculate_openspiel_metanash: bool = scenario.calculate_openspiel_metanash
    calculate_openspiel_metanash_at_end: bool = scenario.calculate_openspiel_metanash_at_end
    calc_metanash_every_n_iters: int = scenario.calc_metanash_every_n_iters
    metrics_smoothing_episodes_override: int = scenario.metanash_metrics_smoothing_episodes_override

    assert scenario["xfdo_metanash_method"] == "nfsp"

    ray_head_address = manager_metadata["ray_head_address"]
    init_ray_for_scenario(scenario=scenario, head_address=ray_head_address, logging_level=logging.INFO)

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
    open_spiel_env_config = tmp_base_env.open_spiel_env_config if calculate_openspiel_metanash else None

    restricted_env_config = {"create_env_fn": _create_base_env, }

    if use_openspiel_restricted_game:
        restricted_game_class = OpenSpielRestrictedGame
        tmp_env = restricted_game_class(env_config=restricted_env_config)
        restricted_game_action_spaces = [tmp_env.base_action_space for _ in range(2)]
    else:
        restricted_game_class = RestrictedGame
        tmp_env = restricted_game_class(env_config=restricted_env_config)
        restricted_game_action_spaces = [Discrete(n=len(player_to_base_game_action_specs[p])) for p in range(2)]

    assert all(restricted_game_action_spaces[0] == space for space in restricted_game_action_spaces)

    avg_trainer_config = merge_dicts({
        "log_level": "DEBUG",
        "framework": "torch",
        "env": restricted_game_class,
        "env_config": restricted_env_config,
        "num_gpus": 0.0,
        "num_gpus_per_worker": 0.0,
        "num_workers": 0,
        "num_envs_per_worker": 1,
        "multiagent": {
            "policies_to_train": ["average_policy_0", "average_policy_1"],
            "policies": {
                "average_policy_0": (
                policy_classes["average_policy"], tmp_env.observation_space, restricted_game_action_spaces[0],
                {"explore": False}),
                "average_policy_1": (
                policy_classes["average_policy"], tmp_env.observation_space, restricted_game_action_spaces[1],
                {"explore": False}),
                "delegate_policy": (
                policy_classes["delegate_policy"], tmp_base_env.observation_space, tmp_env.base_action_space,
                {"explore": False}),
            },
            "policy_mapping_fn": assert_not_called,
        },

    }, get_avg_trainer_config(tmp_env.base_action_space))
    for _policy_id in ["average_policy_0", "average_policy_1"]:
        avg_trainer_config["multiagent"]["policies"][_policy_id][3]["model"] = {
            "custom_model": get_restricted_game_custom_model(tmp_base_env)}

    avg_trainer = avg_trainer_class(config=avg_trainer_config,
                                    logger_creator=get_trainer_logger_creator(
                                        base_dir=results_dir,
                                        scenario_name=f"nfsp_restricted_game_avg_trainer", 
                                        should_log_result_fn=should_log_trainer_result))

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

        def on_episode_end(self, *, worker: "RolloutWorker", base_env: BaseEnv, policies: Dict[PolicyID, Policy],
                           episode: MultiAgentEpisode, env_index: int, **kwargs):
            super().on_episode_end(worker=worker, base_env=base_env, policies=policies, episode=episode,
                                   env_index=env_index, **kwargs)
            episode_policies = set(episode.agent_rewards.keys())
            if episode_policies == {(0, "average_policy_0"), (1, "best_response_1")}:
                worker.avg_br_reward_deque.add.remote(episode.agent_rewards[(1, "best_response_1")])
            elif episode_policies == {(1, "average_policy_1"), (0, "best_response_0")}:
                worker.avg_br_reward_deque.add.remote(episode.agent_rewards[(0, "best_response_0")])

        def on_train_result(self, *, trainer, result: dict, **kwargs):
            super().on_train_result(trainer=trainer, result=result, **kwargs)
            training_iteration = result["training_iteration"]

            result["avg_br_reward_both_players"] = ray.get(trainer.avg_br_reward_deque.get_mean.remote())

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
                    restricted_game_convertors=trainer.get_local_converters(),
                    open_spiel_env_config=open_spiel_env_config,
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
        "env": restricted_game_class,
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
                "average_policy_0": (
                policy_classes["average_policy"], tmp_env.observation_space, restricted_game_action_spaces[0],
                {"explore": False}),
                "best_response_0": (
                policy_classes["best_response"], tmp_env.observation_space, restricted_game_action_spaces[0], {}),

                "average_policy_1": (
                policy_classes["average_policy"], tmp_env.observation_space, restricted_game_action_spaces[1],
                {"explore": False}),
                "best_response_1": (
                policy_classes["best_response"], tmp_env.observation_space, restricted_game_action_spaces[1], {}),

                "delegate_policy": (
                policy_classes["delegate_policy"], tmp_base_env.observation_space, tmp_env.base_action_space,
                {"explore": False}),

            },
            "policy_mapping_fn": select_policy,
        },
    }
    assert all(restricted_game_action_spaces[0] == space for space in restricted_game_action_spaces), \
        "If not true, the line below with \"get_trainer_config\" may need to be changed to a better solution."
    br_trainer_config = merge_dicts(br_trainer_config, get_trainer_config(tmp_env.base_action_space))
    for _policy_id in ["average_policy_0", "average_policy_1", "best_response_0", "best_response_1"]:
        br_trainer_config["multiagent"]["policies"][_policy_id][3]["model"] = {
            "custom_model": get_restricted_game_custom_model(tmp_base_env)}

    br_trainer_config["metrics_smoothing_episodes"] = metrics_smoothing_episodes_override

    br_trainer = trainer_class(config=br_trainer_config,
                               logger_creator=get_trainer_logger_creator(
                                   base_dir=results_dir,
                                   scenario_name="nfsp_restricted_game_trainer",
                                   should_log_result_fn=should_log_trainer_result))

    avg_br_reward_deque = StatDeque.remote(max_items=br_trainer_config["metrics_smoothing_episodes"])

    def _set_avg_br_rew_deque(worker: RolloutWorker):
        worker.avg_br_reward_deque = avg_br_reward_deque

    br_trainer.workers.foreach_worker(_set_avg_br_rew_deque)
    br_trainer.avg_br_reward_deque = avg_br_reward_deque

    if use_openspiel_restricted_game:
        local_delegate_policy = br_trainer.workers.local_worker().policy_map["delegate_policy"]
        player_converters = []
        for p in range(2):
            print("Creating restricted game obs conversions...")
            convertor = get_restricted_game_obs_conversions(player=p, delegate_policy=local_delegate_policy,
                                                            policy_specs=player_to_base_game_action_specs[p],
                                                            load_policy_spec_fn=create_get_pure_strat_cached(cache={}),
                                                            tmp_base_env=tmp_base_env)
            player_converters.append(convertor)
        for _trainer in [br_trainer, avg_trainer]:
            def _set_worker_converters(worker: RolloutWorker):
                worker_delegate_policy = worker.policy_map["delegate_policy"]
                for p in range(2):
                    worker.foreach_env(lambda env: env.set_obs_conversion_dict(p, player_converters[p]))
                worker_delegate_policy.player_converters = player_converters

            _trainer.workers.foreach_worker(_set_worker_converters)
            _trainer.get_local_converters = lambda: _trainer.workers.local_worker().policy_map[
                "delegate_policy"].player_converters
    else:
        weights_cache = {}
        for _trainer in [br_trainer, avg_trainer]:
            def _set_worker_converters(worker: RolloutWorker):
                worker_delegate_policy = worker.policy_map["delegate_policy"]
                player_converters = []
                for p in range(2):
                    player_converter = RestrictedToBaseGameActionSpaceConverter(
                        delegate_policy=worker_delegate_policy, policy_specs=player_to_base_game_action_specs[p],
                        load_policy_spec_fn=create_get_pure_strat_cached(cache=weights_cache))
                    player_converters.append(player_converter)
                    worker.foreach_env(lambda env: env.set_action_conversion(p, player_converter))
                worker_delegate_policy.player_converters = player_converters

            _trainer.workers.foreach_worker(_set_worker_converters)
            _trainer.get_local_converters = lambda: _trainer.workers.local_worker().policy_map[
                "delegate_policy"].player_converters

    br_trainer.latest_avg_trainer_result = None
    train_iter_count = 0

    for _trainer in [br_trainer, avg_trainer]:
        for policy_id, policy in _trainer.workers.local_worker().policy_map.items():
            policy.policy_id = policy_id

    if len(player_to_base_game_action_specs[0]) == 1:
        final_train_result = {"episodes_total": 0, "timesteps_total": 0, "training_iteration": 0}
        tmp_callback = NFSPBestResponseCallbacks()
        tmp_callback.on_train_result(trainer=br_trainer, result=final_train_result)
        if "z_avg_policy_exploitability" in final_train_result:
            print(f"\n\nexploitability: {final_train_result['z_avg_policy_exploitability']}\n\n")
    else:

        avg_weights = avg_trainer.get_weights(["average_policy_0", "average_policy_1"])
        br_trainer.workers.foreach_worker(lambda worker: worker.set_weights(avg_weights))
        while True:
            avg_train_results = avg_trainer.train()
            avg_weights = avg_trainer.get_weights(["average_policy_0", "average_policy_1"])
            br_trainer.workers.foreach_worker(lambda worker: worker.set_weights(avg_weights))
            br_trainer.latest_avg_trainer_result = copy.deepcopy(avg_train_results)
            train_iter_results = br_trainer.train()  # do a step (or several) in the main RL loop

            train_iter_count += 1
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

                if calculate_openspiel_metanash_at_end:
                    openspiel_game_version = base_env_config["version"]
                    local_avg_policy_0 = br_trainer.workers.local_worker().policy_map["average_policy_0"]
                    local_avg_policy_1 = br_trainer.workers.local_worker().policy_map["average_policy_1"]
                    exploitability = xfdo_nfsp_measure_exploitability_nonlstm(
                        rllib_policies=[local_avg_policy_0, local_avg_policy_1],
                        poker_game_version=openspiel_game_version,
                        restricted_game_convertors=br_trainer.get_local_converters(),
                        open_spiel_env_config=open_spiel_env_config,
                    )
                    train_iter_results["z_avg_policy_exploitability"] = exploitability

                final_train_result = deepcopy(train_iter_results)
                break

    avg_policy_specs = []
    for player in range(2):
        strategy_id = f"avg_policy_player_{player}_{datetime_str()}"

        checkpoint_path = save_nfsp_avg_policy_checkpoint(trainer=br_trainer,
                                                          policy_id_to_save=f"average_policy_{player}",
                                                          save_dir=checkpoint_dir(trainer=br_trainer),
                                                          timesteps_training=final_train_result["timesteps_total"],
                                                          episodes_training=final_train_result["episodes_total"],
                                                          checkpoint_name=f"{strategy_id}.h5")

        avg_policy_spec = StrategySpec(
            strategy_id=strategy_id,
            metadata={"checkpoint_path": checkpoint_path,
                      "delegate_policy_specs": [spec.to_json() for spec in player_to_base_game_action_specs[player]]
                      })
        avg_policy_specs.append(avg_policy_spec)

    ray.kill(avg_trainer.workers.local_worker().replay_buffer_actor)
    avg_trainer.cleanup()
    br_trainer.cleanup()
    del avg_trainer
    del br_trainer
    del avg_br_reward_deque

    time.sleep(10)

    assert final_train_result is not None
    return avg_policy_specs, final_train_result
