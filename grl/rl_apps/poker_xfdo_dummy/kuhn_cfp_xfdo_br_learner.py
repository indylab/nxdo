import os
import time
import logging
import random

from gym.spaces import Discrete

from typing import Dict, List
from tables.exceptions import HDF5ExtError
import deepdish
import argparse
import ray



from ray.rllib.agents import Trainer
from ray.rllib.agents.sac import SACTrainer, SACTorchPolicy
from ray.rllib.agents.dqn import DQNTrainer, DQNTorchPolicy, SimpleQTorchPolicy

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
from ray.rllib.utils.typing import AgentID, PolicyID
from grl.p2sro.payoff_table import PayoffTableStrategySpec
from grl.rl_apps.kuhn_poker_p2sro.poker_multi_agent_env import PokerMultiAgentEnv
from grl.utils import pretty_dict_str, datetime_str, ensure_dir
from grl.p2sro.p2sro_manager.utils import PolicySpecDistribution
from grl.xfdo.xfdo_manager.remote import RemoteXFDOManagerClient
from grl.rl_apps.kuhn_poker_p2sro.config import kuhn_sac_params, kuhn_dqn_params
from grl.xfdo.action_space_conversion import RestrictedToBaseGameActionSpaceConverter
from grl.xfdo.restricted_game import RestrictedGame

logger = logging.getLogger(__name__)

torch, _ = try_import_torch()

BR_CHECKPOINT_SAVE_DIR = "/tmp/p2sro_policies"


def save_best_response_checkpoint(trainer: SACTrainer,
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
    br_weights = {k.replace(".", "_dot_"): v for k, v in br_weights.items()} # periods cause HDF5 NaturalNaming warnings
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
            if attempt+1 == num_save_attempts:
                raise
            time.sleep(1.0)
    return checkpoint_path


def load_pure_strat(policy: Policy, pure_strat_spec, checkpoint_path: str = None):
    assert pure_strat_spec is None or checkpoint_path is None, "can only pass one or the other"
    if checkpoint_path is None:
        if hasattr(policy, "policy_spec") and pure_strat_spec == policy.policy_spec:
            return
        pure_strat_checkpoint_path = pure_strat_spec.metadata["checkpoint_path"]
    else:
        pure_strat_checkpoint_path = checkpoint_path
    checkpoint_data = deepdish.io.load(path=pure_strat_checkpoint_path)
    weights = checkpoint_data["weights"]
    weights = {k.replace("_dot_", "."): v for k, v in weights.items()}
    policy.set_weights(weights=weights)
    policy.policy_spec = pure_strat_spec


def create_get_pure_strat_cached(cache: dict):
    def load_pure_strat_cached(policy: Policy, pure_strat_spec):

        pure_strat_checkpoint_path = pure_strat_spec.metadata["checkpoint_path"]

        if pure_strat_checkpoint_path in cache:
            weights = cache[pure_strat_checkpoint_path]
        else:
            checkpoint_data = deepdish.io.load(path=pure_strat_checkpoint_path)
            weights = checkpoint_data["weights"]
            weights = {k.replace("_dot_", "."): v for k, v in weights.items()}
            cache[pure_strat_checkpoint_path] = weights

        policy.set_weights(weights=weights)
        policy.policy_spec = pure_strat_spec
    return load_pure_strat_cached


def create_metadata_with_new_checkpoint_for_current_best_response(trainer: SACTrainer,
                                                                  player: int,
                                                              save_dir: str,
                                                              timesteps_training_br: int,
                                                              episodes_training_br: int,
                                                              active_policy_num: int = None,
                                                              ):
    return {
        "checkpoint_path": save_best_response_checkpoint(trainer=trainer,
                                                         player=player,
                                                         save_dir=save_dir,
                                                         active_policy_num=active_policy_num,
                                                         timesteps_training_br=timesteps_training_br,
                                                         episodes_training_br=episodes_training_br),
        "timesteps_training_br": timesteps_training_br,
        "episodes_training_br": episodes_training_br
    }


def set_best_response_active_policy_spec_and_player_for_all_workers(trainer: SACTrainer,
                                                                    player: int,
                                                         active_policy_spec: PayoffTableStrategySpec):
    def _set_p2sro_policy_spec_on_best_response_policy(worker: RolloutWorker):
        br_policy: SACTorchPolicy = worker.policy_map[f"best_response"]
        br_policy.p2sro_policy_spec = active_policy_spec
        worker.br_player = player
    trainer.workers.foreach_worker(_set_p2sro_policy_spec_on_best_response_policy)


class P2SROPreAndPostEpisodeCallbacks(DefaultCallbacks):

    def on_episode_start(self, *, worker: "RolloutWorker", base_env: BaseEnv, policies: Dict[PolicyID, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        super().on_episode_start(worker=worker, base_env=base_env, policies=policies, episode=episode,
                                 env_index=env_index, **kwargs)
        avg_policy: Policy = worker.policy_map["metanash"]
        if not hasattr(avg_policy, "br_specs") or not avg_policy.br_specs:
            return
        new_br_spec = random.choice(avg_policy.br_specs)
        if hasattr(avg_policy, "policy_spec") and new_br_spec == avg_policy.policy_spec:
            return
        load_pure_strat(policy=avg_policy, pure_strat_spec=new_br_spec)


def set_restricted_game_conversations_for_all_workers(
        trainer: Trainer,
        delegate_policy_id: PolicyID,
        agent_id_to_restricted_game_specs: Dict[AgentID, List[PayoffTableStrategySpec]],
        load_policy_spec_fn):

    def _set_conversions(worker: RolloutWorker):

        def _set_restricted_env_convertions(restricted_env):
            assert isinstance(restricted_env, RestrictedGame)
            for agent_id, action_policy_specs in agent_id_to_restricted_game_specs.items():
                if len(action_policy_specs) > 0:
                    convertor = RestrictedToBaseGameActionSpaceConverter(
                        delegate_policy=worker.policy_map[delegate_policy_id],
                        policy_specs=action_policy_specs,
                        load_policy_spec_fn=load_pure_strat)
                    restricted_env.set_action_conversion(agent_id=agent_id, converter=convertor)
        worker.foreach_env(_set_restricted_env_convertions)
    trainer.workers.foreach_worker(_set_conversions)


def train_poker_best_response(br_player: int, print_train_results: bool = True):

    xfdo_manager = RemoteXFDOManagerClient(n_players=2, port=os.getenv("XFDO_PORT", 4546), remote_server_host="127.0.0.1")
    br_params = xfdo_manager.claim_new_active_policy_for_player(player=br_player)
    metanash_specs_for_players, delegate_specs_for_players, active_policy_num = br_params

    if metanash_specs_for_players:
        cfp_metanash_specs_for_players = {}
        for p, player_metanash_spec in metanash_specs_for_players.items():
            player_cfp_json_specs = player_metanash_spec.metadata["cfp_pure_strat_specs"]
            player_cfp_specs = [PayoffTableStrategySpec.from_json(json_spec) for json_spec in player_cfp_json_specs]
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

    base_env_config = {'version': "kuhn_poker", "fixed_players": True, "dummy_action_multiplier": 20}
    restricted_env_config = {
        "create_env_fn": lambda: PokerMultiAgentEnv(env_config=base_env_config),
        "raise_if_no_restricted_players": cfp_metanash_specs_for_players is not None
    }

    tmp_env = RestrictedGame(env_config=restricted_env_config)

    if cfp_metanash_specs_for_players:
        other_player_restricted_action_space = Discrete(n=len(delegate_specs_for_players[other_player]))
        # input(f"other_player_restricted_action_space: {other_player_restricted_action_space}")
        # input(f"avg_policy_specs: {[spec.to_json() for spec in metanash_specs_for_players.values()]}")
    else:
        other_player_restricted_action_space = tmp_env.base_action_space

    trainer_config = {
        "callbacks": P2SROPreAndPostEpisodeCallbacks,
        "env": RestrictedGame,
        "env_config": restricted_env_config,
        "gamma": 1.0,
        "num_gpus": 0,
        "num_workers": 0,
        "num_envs_per_worker": 1,
        "multiagent": {
            "policies_to_train": [f"best_response"],
            "policies": {
                f"metanash": (SimpleQTorchPolicy, tmp_env.observation_space, other_player_restricted_action_space, {"explore": False}),
                f"metanash_delegate": (SimpleQTorchPolicy, tmp_env.observation_space, tmp_env.base_action_space, {"explore": False}),
                f"best_response": (SimpleQTorchPolicy, tmp_env.observation_space, tmp_env.base_action_space, {}),
            },
            "policy_mapping_fn": select_policy,
        },
    }
    trainer_config = merge_dicts(trainer_config, kuhn_dqn_params(action_space=tmp_env.base_action_space))

    ray.init(ignore_reinit_error=True, local_mode=False)
    trainer = DQNTrainer(config=trainer_config)

    if cfp_metanash_specs_for_players:
        def _set_worker_metanash_cfp_specs(worker: RolloutWorker):
            worker.policy_map["metanash"].br_specs = cfp_metanash_specs_for_players[other_player]
        trainer.workers.foreach_worker(_set_worker_metanash_cfp_specs)

    trainer.weights_cache = {}
    if delegate_specs_for_players:
        set_restricted_game_conversations_for_all_workers(trainer=trainer, delegate_policy_id="metanash_delegate",
                                                          agent_id_to_restricted_game_specs={
                                                              other_player: delegate_specs_for_players[other_player]},
                                                          load_policy_spec_fn=create_get_pure_strat_cached(cache=trainer.weights_cache))

    log(f"got policy {active_policy_num}")

    # Perform main RL training loop. Stop if we reach max iters or saturate.
    # Saturation is determined by checking if we improve by a minimum amount every n iters.
    train_iter_count = 0

    dont_do_saturation_checks_before_n_train_iters = 300
    iters_since_saturation_checks_began = None
    check_for_saturation_every_n_train_iters = 100
    minimum_reward_improvement_otherwise_saturated = 0.01
    last_saturation_check_reward = None

    max_train_iters = 20000

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
            log(pretty_dict_str(train_iter_results))

        total_timesteps_training_br = train_iter_results["timesteps_total"]
        total_episodes_training_br = train_iter_results["episodes_total"]
        br_reward_this_iter = train_iter_results["policy_reward_mean"][f"best_response"]

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
            break

    log(f"Training stopped. Setting active policy {active_policy_num} as fixed.")

    xfdo_manager.submit_final_br_policy(
        player=br_player, policy_num=active_policy_num,
        metadata_dict=create_metadata_with_new_checkpoint_for_current_best_response(
            trainer=trainer, player=br_player, save_dir=BR_CHECKPOINT_SAVE_DIR, timesteps_training_br=total_timesteps_training_br,
            episodes_training_br=total_episodes_training_br,
            active_policy_num=active_policy_num
        ))

    # wait for both player policies to be fixed and then track exploitability.
    for player_to_wait_on in range(2):
        wait_count = 0
        while True:
            if xfdo_manager.is_policy_fixed(player=player_to_wait_on, policy_num=active_policy_num):
                break
            if wait_count % 10 == 0:
                log(f"Waiting for policy {active_policy_num} player {player_to_wait_on} to become fixed")
            time.sleep(2.0)
            wait_count += 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--player', type=int)
    args = parser.parse_args()

    while True:
        # Train a br for each player, then repeat.
        train_poker_best_response(br_player=args.player, print_train_results=True)


        

