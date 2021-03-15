import ray
from ray.rllib.utils import merge_dicts, try_import_torch

torch, _ = try_import_torch()

import os
import time
import logging
import random
import tempfile
from gym.spaces import Discrete

from typing import Dict, List, Type
from tables.exceptions import HDF5ExtError
import deepdish
import argparse

from ray.rllib.agents import Trainer
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import AgentID, PolicyID
from grl.utils.strategy_spec import StrategySpec
from grl.utils.common import pretty_dict_str, datetime_str, ensure_dir
from grl.algos.nxdo.nxdo_manager.remote import RemoteNXDOManagerClient
from grl.algos.nxdo.action_space_conversion import RestrictedToBaseGameActionSpaceConverter
from grl.algos.nxdo.restricted_game import RestrictedGame
from grl.rllib_tools.space_saving_logger import SpaceSavingLogger, get_trainer_logger_creator
from grl.rllib_tools.policy_checkpoints import load_pure_strat, save_policy_checkpoint, create_get_pure_strat_cached
from grl.rl_apps.scenarios.catalog import scenario_catalog
from grl.rl_apps.scenarios.nxdo_scenario import NXDOScenario
from grl.rl_apps.scenarios.ray_setup import init_ray_for_scenario
from grl.rl_apps.scenarios.stopping_conditions import StoppingCondition
from grl.algos.nxdo.opnsl_restricted_game import OpenSpielRestrictedGame, get_restricted_game_obs_conversions
from grl.utils.port_listings import get_client_port_for_service
from grl.rl_apps import GRL_SEED

logger = logging.getLogger(__name__)


def checkpoint_dir(trainer):
    return os.path.join(trainer.logdir, "br_checkpoints")


def save_best_response_checkpoint(trainer,
                                  player: int,
                                  save_dir: str,
                                  timesteps_training_br: int,
                                  episodes_training_br: int,
                                  active_policy_num: int = None,
                                  average_br_reward: float = None):
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
                "episodes_training_br": episodes_training_br,
                "average_br_reward": average_br_reward,
            })
            break
        except HDF5ExtError:
            if attempt + 1 == num_save_attempts:
                raise
            time.sleep(1.0)
    return checkpoint_path


def create_metadata_with_new_checkpoint_for_current_best_response(trainer,
                                                                  player: int,
                                                                  save_dir: str,
                                                                  timesteps_training_br: int,
                                                                  episodes_training_br: int,
                                                                  active_policy_num: int = None,
                                                                  average_br_reward: float = None
                                                                  ):
    return {
        "checkpoint_path": save_best_response_checkpoint(trainer=trainer,
                                                         player=player,
                                                         save_dir=save_dir,
                                                         active_policy_num=active_policy_num,
                                                         timesteps_training_br=timesteps_training_br,
                                                         episodes_training_br=episodes_training_br),
        "timesteps_training_br": timesteps_training_br,
        "episodes_training_br": episodes_training_br,
        "average_br_reward": average_br_reward,
    }


def set_best_response_active_policy_spec_and_player_for_all_workers(trainer,
                                                                    player: int,
                                                                    active_policy_spec: StrategySpec):
    def _set_p2sro_policy_spec_on_best_response_policy(worker: RolloutWorker):
        br_policy = worker.policy_map[f"best_response"]
        br_policy.p2sro_policy_spec = active_policy_spec
        worker.br_player = player

    trainer.workers.foreach_worker(_set_p2sro_policy_spec_on_best_response_policy)


class P2SROPreAndPostEpisodeCallbacks(DefaultCallbacks):

    def on_episode_start(self, *, worker: "RolloutWorker", base_env: BaseEnv, policies: Dict[PolicyID, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        super().on_episode_start(worker=worker, base_env=base_env, policies=policies, episode=episode,
                                 env_index=env_index, **kwargs)
        avg_policy: Policy = worker.policy_map["metanash"]
        if not hasattr(avg_policy, "cfp_br_specs") or not avg_policy.cfp_br_specs:
            return
        new_br_spec = random.choice(avg_policy.cfp_br_specs)
        if hasattr(avg_policy, "policy_spec") and new_br_spec == avg_policy.policy_spec:
            return
        load_pure_strat(policy=avg_policy, pure_strat_spec=new_br_spec)


def set_restricted_game_conversations_for_all_workers(
        trainer: Trainer,
        delegate_policy_id: PolicyID,
        agent_id_to_restricted_game_specs: Dict[AgentID, List[StrategySpec]],
        load_policy_spec_fn):
    def _set_conversions(worker: RolloutWorker):

        def _set_restricted_env_convertions(restricted_env):
            assert isinstance(restricted_env, RestrictedGame)
            for agent_id, action_policy_specs in agent_id_to_restricted_game_specs.items():
                if len(action_policy_specs) > 0:
                    convertor = RestrictedToBaseGameActionSpaceConverter(
                        delegate_policy=worker.policy_map[delegate_policy_id],
                        policy_specs=action_policy_specs,
                        load_policy_spec_fn=load_policy_spec_fn)
                    restricted_env.set_action_conversion(agent_id=agent_id, converter=convertor)

        worker.foreach_env(_set_restricted_env_convertions)

    trainer.workers.foreach_worker(_set_conversions)


def set_restricted_game_conversions_for_all_workers_openspiel(
        trainer: Trainer,
        tmp_base_env: MultiAgentEnv,
        delegate_policy_id: PolicyID,
        agent_id_to_restricted_game_specs: Dict[AgentID, List[StrategySpec]],
        load_policy_spec_fn):
    local_delegate_policy = trainer.workers.local_worker().policy_map[delegate_policy_id]
    player_converters = {}
    for p, restricted_game_specs in agent_id_to_restricted_game_specs.items():
        if len(restricted_game_specs) == 0:
            continue
        player_converters[p] = (get_restricted_game_obs_conversions(player=p, delegate_policy=local_delegate_policy,
                                                                    policy_specs=restricted_game_specs,
                                                                    load_policy_spec_fn=load_policy_spec_fn,
                                                                    tmp_base_env=tmp_base_env))
    assert len(player_converters) == 0 or len(player_converters) == 1

    def _set_worker_converters(worker: RolloutWorker):
        worker_delegate_policy = worker.policy_map[delegate_policy_id]
        for p, player_converter in player_converters.items():
            worker.foreach_env(lambda env: env.set_obs_conversion_dict(p, player_converter))
        worker_delegate_policy.player_converters = player_converters

    trainer.workers.foreach_worker(_set_worker_converters)
    trainer.get_local_converters = lambda: trainer.workers.local_worker().policy_map[
        delegate_policy_id].player_converters


def train_nxdo_best_response(br_player: int,
                             scenario_name: str,
                             nxdo_manager_port: int,
                             nxdo_manager_host: str,
                             print_train_results: bool = True):
    scenario: NXDOScenario = scenario_catalog.get(scenario_name=scenario_name)

    use_openspiel_restricted_game: bool = scenario.use_openspiel_restricted_game
    get_restricted_game_custom_model = scenario.get_restricted_game_custom_model

    env_class = scenario.env_class
    base_env_config = scenario.env_config
    trainer_class = scenario.trainer_class_br
    policy_classes: Dict[str, Type[Policy]] = scenario.policy_classes_br
    get_trainer_config = scenario.get_trainer_config_br
    nxdo_br_get_stopping_condition = scenario.get_stopping_condition_br
    nxdo_metanash_method: str = scenario.xdo_metanash_method
    use_cfp_metanash = (nxdo_metanash_method == "cfp")

    nxdo_manager = RemoteNXDOManagerClient(n_players=2,
                                           port=nxdo_manager_port,
                                           remote_server_host=nxdo_manager_host)

    manager_metadata = nxdo_manager.get_manager_metadata()
    results_dir = nxdo_manager.get_log_dir()

    br_params = nxdo_manager.claim_new_active_policy_for_player(player=br_player)
    metanash_specs_for_players, delegate_specs_for_players, active_policy_num = br_params

    if metanash_specs_for_players is not None and use_cfp_metanash:
        cfp_metanash_specs_for_players = {}
        for p, player_metanash_spec in metanash_specs_for_players.items():
            player_cfp_json_specs = player_metanash_spec.metadata["cfp_pure_strat_specs"]
            player_cfp_specs = [StrategySpec.from_json(json_spec) for json_spec in player_cfp_json_specs]
            cfp_metanash_specs_for_players[p] = player_cfp_specs
    else:
        cfp_metanash_specs_for_players = None

    other_player = 1 - br_player
    br_learner_name = f"policy {active_policy_num} player {br_player}"

    def log(message, level=logging.INFO):
        logger.log(level, f"({br_learner_name}): {message}")

    def select_policy(agent_id):
        if agent_id == br_player:
            return f"best_response"
        elif agent_id == other_player:
            return f"metanash"
        else:
            raise ValueError(f"Unknown agent id: {agent_id}")

    restricted_env_config = {
        "create_env_fn": lambda: env_class(env_config=base_env_config),
        "raise_if_no_restricted_players": metanash_specs_for_players is not None
    }
    tmp_base_eny = env_class(env_config=base_env_config)

    if use_openspiel_restricted_game:
        restricted_game_class = OpenSpielRestrictedGame
    else:
        restricted_game_class = RestrictedGame
        restricted_env_config["use_delegate_policy_exploration"] = scenario.allow_stochastic_best_responses

    tmp_env = restricted_game_class(env_config=restricted_env_config)

    if metanash_specs_for_players is None or use_openspiel_restricted_game:
        other_player_restricted_action_space = tmp_env.base_action_space
        metanash_class = policy_classes["best_response"]
    else:
        other_player_restricted_action_space = Discrete(n=len(delegate_specs_for_players[other_player]))
        metanash_class = policy_classes["metanash"]
        print(
            f"metanash class: {metanash_class}, other_player_restricted_action_space: {other_player_restricted_action_space}")

    if metanash_specs_for_players is None and use_openspiel_restricted_game:
        other_player_restricted_obs_space = tmp_env.base_observation_space
    else:
        other_player_restricted_obs_space = tmp_env.observation_space

    trainer_config = {
        "callbacks": P2SROPreAndPostEpisodeCallbacks,
        "env": restricted_game_class,
        "env_config": restricted_env_config,
        "gamma": 1.0,
        "num_gpus": 0,
        "num_workers": 0,
        "num_envs_per_worker": 1,
        "multiagent": {
            "policies_to_train": [f"best_response"],
            "policies": {
                f"metanash": (metanash_class, other_player_restricted_obs_space, other_player_restricted_action_space,
                              {"explore": False}),
                f"metanash_delegate": (
                policy_classes["best_response"], tmp_env.base_observation_space, tmp_env.base_action_space,
                {"explore": scenario.allow_stochastic_best_responses}),
                f"best_response": (
                policy_classes["best_response"], tmp_env.base_observation_space, tmp_env.base_action_space, {}),
            },
            "policy_mapping_fn": select_policy,
        },
    }

    if metanash_specs_for_players is not None and get_restricted_game_custom_model is not None:
        trainer_config["multiagent"]["policies"]["metanash"][3]["model"] = {
            "custom_model": get_restricted_game_custom_model(tmp_base_eny)}

    trainer_config = merge_dicts(trainer_config, get_trainer_config(tmp_env))

    ray_head_address = manager_metadata["ray_head_address"]
    init_ray_for_scenario(scenario=scenario, head_address=ray_head_address, logging_level=logging.INFO)

    trainer = trainer_class(config=trainer_config, logger_creator=get_trainer_logger_creator(
        base_dir=results_dir, scenario_name=scenario_name,
        should_log_result_fn=lambda result: result["training_iteration"] % 100 == 0))

    if use_cfp_metanash and cfp_metanash_specs_for_players:
        # metanash is uniform distribution of pure strat specs
        def _set_worker_metanash_cfp_specs(worker: RolloutWorker):
            worker.policy_map["metanash"].cfp_br_specs = cfp_metanash_specs_for_players[other_player]

        trainer.workers.foreach_worker(_set_worker_metanash_cfp_specs)
    elif not use_cfp_metanash:
        # metanash is single pure strat spec
        def _set_worker_metanash(worker: RolloutWorker):
            if metanash_specs_for_players is not None:
                metanash_policy = worker.policy_map["metanash"]
                load_pure_strat(policy=metanash_policy, pure_strat_spec=metanash_specs_for_players[other_player])

        trainer.workers.foreach_worker(_set_worker_metanash)

    trainer.weights_cache = {}
    if delegate_specs_for_players:
        if use_openspiel_restricted_game:
            set_restricted_game_conversions_for_all_workers_openspiel(trainer=trainer,
                                                                      tmp_base_env=tmp_base_eny,
                                                                      delegate_policy_id="metanash_delegate",
                                                                      agent_id_to_restricted_game_specs={
                                                                          other_player: delegate_specs_for_players[
                                                                              other_player]},
                                                                      load_policy_spec_fn=load_pure_strat)
        else:
            set_restricted_game_conversations_for_all_workers(trainer=trainer, delegate_policy_id="metanash_delegate",
                                                              agent_id_to_restricted_game_specs={
                                                                  other_player: delegate_specs_for_players[
                                                                      other_player]},
                                                              load_policy_spec_fn=create_get_pure_strat_cached(
                                                                  cache=trainer.weights_cache))

    log(f"got policy {active_policy_num}")

    # Perform main RL training loop. Stop if we reach max iters or saturate.
    # Saturation is determined by checking if we improve by a minimum amount every n iters.
    train_iter_count = 0

    stopping_condition: StoppingCondition = nxdo_br_get_stopping_condition()

    while True:
        train_iter_results = trainer.train()  # do a step (or several) in the main RL loop
        train_iter_count += 1

        if print_train_results:
            train_iter_results["p2sro_active_policy_num"] = active_policy_num
            train_iter_results["best_response_player"] = br_player
            # Delete verbose debugging info before printing
            if "hist_stats" in train_iter_results:
                del train_iter_results["hist_stats"]
            if "td_error" in train_iter_results["info"]["learner"][f"best_response"]:
                del train_iter_results["info"]["learner"][f"best_response"]["td_error"]
            log(f"Trainer log dir is {trainer.logdir}")
            log(pretty_dict_str(train_iter_results))

        total_timesteps_training_br = train_iter_results["timesteps_total"]
        total_episodes_training_br = train_iter_results["episodes_total"]
        br_reward_this_iter = train_iter_results["policy_reward_mean"][f"best_response"]

        if stopping_condition.should_stop_this_iter(latest_trainer_result=train_iter_results):
            log("Stopping condition met.")
            break

    log(f"Training stopped. Setting active policy {active_policy_num} as fixed.")

    nxdo_manager.submit_final_br_policy(
        player=br_player, policy_num=active_policy_num,
        metadata_dict=create_metadata_with_new_checkpoint_for_current_best_response(
            trainer=trainer, player=br_player, save_dir=checkpoint_dir(trainer=trainer),
            timesteps_training_br=total_timesteps_training_br,
            episodes_training_br=total_episodes_training_br,
            active_policy_num=active_policy_num,
            average_br_reward=float(br_reward_this_iter),
        ))

    # trainer.cleanup()
    # del trainer
    ray.shutdown()
    time.sleep(10)

    # wait for both player policies to be fixed.
    for player_to_wait_on in range(2):
        wait_count = 0
        while True:
            if nxdo_manager.is_policy_fixed(player=player_to_wait_on, policy_num=active_policy_num):
                break
            if wait_count % 10 == 0:
                log(f"Waiting for policy {active_policy_num} player {player_to_wait_on} to become fixed")
            time.sleep(2.0)
            wait_count += 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--player', type=int)
    parser.add_argument('--scenario', type=str)
    parser.add_argument('--nxdo_port', type=int, required=False, default=None)
    parser.add_argument('--nxdo_host', type=str, required=False, default='localhost')
    commandline_args = parser.parse_args()

    scenario_name = commandline_args.scenario

    nxdo_host = commandline_args.nxdo_host
    nxdo_port = commandline_args.nxdo_port
    if nxdo_port is None:
        nxdo_port = get_client_port_for_service(service_name=f"seed_{GRL_SEED}_{scenario_name}")

    while True:
        # Train a br for each player, then repeat.
        train_nxdo_best_response(
            br_player=commandline_args.player,
            scenario_name=scenario_name,
            nxdo_manager_port=nxdo_port,
            nxdo_manager_host=nxdo_host,
            print_train_results=True,
        )
