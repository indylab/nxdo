import os
import time
import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Callable, Type, Dict, Union
import tempfile
import argparse
import random
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, Future

from collections import defaultdict
from gym.spaces import Space, Discrete

import deepdish

import ray
from ray.rllib.utils import merge_dicts, try_import_torch
torch, _ = try_import_torch()

from ray.rllib import SampleBatch, Policy
from ray.rllib.agents import Trainer
from ray.rllib.agents.sac import SACTrainer, SACTorchPolicy
from ray.rllib.agents.dqn import DQNTrainer, DQNTorchPolicy, SimpleQTorchPolicy, SimpleQTFPolicy
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
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchCategorical, \
    TorchDistributionWrapper

import grl
from grl.utils import pretty_dict_str, datetime_str, ensure_dir, copy_attributes


from grl.p2sro.payoff_table import PayoffTableStrategySpec
from grl.rl_apps.nfsp.config import kuhn_dqn_params
from grl.rl_apps.poker_xfdo.poker_utils import xfdo_snfsp_measure_exploitability_nonlstm

from grl.rllib_tools.space_saving_logger import SpaceSavingLogger
from grl.xfdo.restricted_game import RestrictedGame
from grl.xfdo.action_space_conversion import RestrictedToBaseGameActionSpaceConverter

from grl.rl_apps.kuhn_poker_p2sro.poker_multi_agent_env import PokerMultiAgentEnv


logger = logging.getLogger(__name__)


class ReservoirBuffer(object):
    """Allows uniform sampling over a stream of data.

    This class supports the storage of arbitrary elements, such as observation
    tensors, integer actions, etc.

    See https://en.wikipedia.org/wiki/Reservoir_sampling for more details.
    """

    def __init__(self, reservoir_buffer_capacity=500):
        self._reservoir_buffer_capacity = reservoir_buffer_capacity
        self._data = []
        self._add_calls = 0

    def ask_to_add(self):
        self._add_calls += 1
        if len(self._data) < self._reservoir_buffer_capacity:
            return True, None
        else:
            idx = np.random.randint(0, self._add_calls)
            if idx < self._reservoir_buffer_capacity:
                return True, idx
        return False, None

    def add(self, element, idx):
        """Potentially adds `element` to the reservoir buffer.

        Args:
          element: data to be added to the reservoir buffer.
        """
        if idx is None:
            assert len(self._data) < self._reservoir_buffer_capacity
            self._data.append(element)
        else:
            assert idx < self._reservoir_buffer_capacity
            self._data[idx] = element

    def sample(self, num_samples=1):
        """Returns `num_samples` uniformly sampled from the buffer.

        Args:
          num_samples: `int`, number of samples to draw.

        Returns:
          An iterable over `num_samples` random elements of the buffer.

        Raises:
          ValueError: If there are less than `num_samples` elements in the buffer
        """
        if len(self._data) < num_samples:
            raise ValueError("{} elements could not be sampled from size {}".format(
                num_samples, len(self._data)))
        return random.sample(self._data, num_samples)

    def to_list(self):
        return list(self._data)

    def __setitem__(self, index, value):
        self._data[index] = value

    def __getitem__(self, index):
        return self._data[index]

    def clear(self):
        self._data = []
        self._add_calls = 0

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)


def save_snfsp_best_response_checkpoint(trainer: Trainer,
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



def create_metadata_with_new_checkpoint_for_current_best_response(br_trainer: Trainer,
                                                                policy_id_to_save: str,
                                                              save_dir: str,
                                                              timesteps_training_br: int,
                                                              episodes_training_br: int,
                                                              checkpoint_name=None
                                                              ):
    return {
        "checkpoint_path": save_snfsp_best_response_checkpoint(trainer=br_trainer,
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


def get_trainer_logger_creator(base_dir: str, env_class):
    logdir_prefix = f"{env_class.__name__}_sparse_{datetime_str()}"

    def trainer_logger_creator(config):
        """Creates a Unified logger with a default logdir prefix
        containing the agent name and the env id
        """
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        logdir = tempfile.mkdtemp(
            prefix=logdir_prefix, dir=base_dir)

        def _should_log(result: dict) -> bool:
            # return "z_avg_policy_exploitability" in result
            return True

        return SpaceSavingLogger(config=config, logdir=logdir, should_log_result_fn=_should_log)

    return trainer_logger_creator


def checkpoint_dir(trainer: Trainer):
    return os.path.join(trainer.logdir, "br_checkpoints")


class ParallelSNFSPBestResponseCallbacks(DefaultCallbacks):

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        super().on_train_result(trainer=trainer, result=result, **kwargs)
        # print(trainer.latest_avg_trainer_result.keys())
        # log(pretty_dict_str(trainer.latest_avg_trainer_result))
        # if trainer.latest_avg_trainer_result is not None:
        #     result["avg_trainer_info"] = trainer.latest_avg_trainer_result.get("info", {})

        if "hist_stats" in result:
            del result["hist_stats"]
        if "info" in result and "td_error" in result["info"]["learner"]["best_response"]:
            del result["info"]["learner"]["best_response"]["td_error"]

        result["br_player"] = trainer.player

        if trainer.player == 0:
            result["other_trainer_results"] = trainer.other_player_latest_train_result
            if len(trainer.other_player_latest_train_result) > 0:
                result["timesteps_total"] += trainer.other_player_latest_train_result["timesteps_total"]
                result["episodes_total"] += trainer.other_player_latest_train_result["episodes_total"]

                br_0_rew = result["policy_reward_mean"]["best_response"]
                br_1_rew = result["other_trainer_results"]["policy_reward_mean"]["best_response"]
                avg_br_reward = (br_0_rew + br_1_rew) / 2.0
                result["avg_br_reward_both_players"] = avg_br_reward

            br_trainer_0 = trainer
            br_trainer_1 = trainer.other_trainer

            training_iteration = result["training_iteration"]
            if training_iteration == 1 or training_iteration % 1000 == 0:
                local_avg_policy_0 = br_trainer_1.workers.local_worker().policy_map["opponent_average_policy"]
                local_avg_policy_1 = br_trainer_0.workers.local_worker().policy_map["opponent_average_policy"]

                br_checkpoint_path_tuple_list: List[Tuple[str, str]] = []
                for br_spec_0, br_spec_1 in zip(local_avg_policy_0.br_specs, local_avg_policy_1.br_specs):
                    br_checkpoint_path_tuple_list.append((
                        br_spec_0.metadata["checkpoint_path"],
                        br_spec_1.metadata["checkpoint_path"]
                    ))

                exploitability = xfdo_snfsp_measure_exploitability_nonlstm(
                    br_checkpoint_path_tuple_list=br_checkpoint_path_tuple_list,
                    set_policy_weights_fn=lambda policy, path: load_pure_strat(policy=policy, checkpoint_path=path, pure_strat_spec=None),
                    rllib_policies=[local_avg_policy_0, local_avg_policy_1],
                    poker_game_version="kuhn_poker",
                    action_space_converters=trainer.get_local_converters(),
                )
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

def train_poker_off_policy_rl_nfsp(results_dir: str,
                                   br_trainer_class: Type[Trainer],
                                   br_policy_class: Type[Policy],
                                   hyperparams: Callable[[Space], Dict],
                                   player_to_base_game_action_specs: Dict[int, List[PayoffTableStrategySpec]],
                                   print_train_results: bool = True):

    ray.init(ignore_reinit_error=True)

    def log(message, level=logging.INFO):
        logger.log(level, message)

    def get_select_policy(trainer_player_num: int):
        def select_policy(agent_id: int):
            if agent_id == trainer_player_num:
                return "best_response"
            elif agent_id == 1 - trainer_player_num:
                if np.random.random() < 0.1:
                    return "opponent_best_response"
                else:
                    return "opponent_average_policy"
            else:
                raise ValueError(f"unexpected agent_id: {agent_id}")
        return select_policy

    def _create_base_env():
        return PokerMultiAgentEnv(env_config={
            'version': "kuhn_poker",
            'fixed_players': True,
            "dummy_action_multiplier": 20
        })

    tmp_base_env = _create_base_env()

    env_config = {"create_env_fn": _create_base_env}
    tmp_env = RestrictedGame(env_config=env_config)
    restricted_game_action_spaces = [Discrete(n=len(player_to_base_game_action_specs[p])) for p in range(2)]

    assert all(restricted_game_action_spaces[0] == space for space in restricted_game_action_spaces)

    br_trainer_configs = [merge_dicts({
        "log_level": "DEBUG",
        "callbacks": ParallelSNFSPBestResponseCallbacks,
        "env": RestrictedGame,
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
                "best_response": (br_policy_class, tmp_env.observation_space, restricted_game_action_spaces[player], {}),
                "opponent_average_policy": (br_policy_class, tmp_env.observation_space, restricted_game_action_spaces[1 - player], {}),
                "opponent_best_response": (br_policy_class, tmp_env.observation_space, restricted_game_action_spaces[1 - player], {}),
                "delegate_policy": (br_policy_class, tmp_base_env.observation_space, tmp_env.base_action_space, {}),
            },
            "policy_mapping_fn": get_select_policy(trainer_player_num=player),
        },
    }, hyperparams(restricted_game_action_spaces[player])) for player in range(2)]

    plateau_reward_episode_window = int(50000)
    for config in br_trainer_configs:
        config["metrics_smoothing_episodes"] = plateau_reward_episode_window

    br_trainers: List[Trainer] = [br_trainer_class(config=br_trainer_configs[player],
                                    logger_creator=get_trainer_logger_creator(base_dir=results_dir, env_class=PokerMultiAgentEnv))
                    for player in range(2)]

    weights_cache = {}
    for trainer in br_trainers:
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

        trainer.workers.foreach_worker(_set_worker_converters)
        trainer.get_local_converters = lambda: trainer.workers.local_worker().policy_map["delegate_policy"].player_converters

    for player, br_trainer in enumerate(br_trainers):
        other_player = 1-player
        br_trainer.player = player
        br_trainer.other_trainer = br_trainers[other_player]
        br_trainer.other_player_latest_train_result = {}

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
                local_avg_policy.br_specs = ReservoirBuffer()
            can_add, add_idx = local_avg_policy.br_specs.ask_to_add()
            if can_add:
                if add_idx is not None:
                    # delete the checkpoint that we're replacing
                    old_spec = local_avg_policy.br_specs[add_idx]
                    checkpoint_path = old_spec.metadata["checkpoint_path"]
                    os.remove(checkpoint_path)

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

    train_iter_count = 0
    checkpoint_count = 0
    checkpoint_every_n_iters = 1
    print("starting")
    checkpoint_brs_for_average_policy(checkpoint_count=checkpoint_count, both_train_iter_results=(None, None))

    if restricted_game_action_spaces[0].n == 1:
        final_train_result = {"episodes_total": 0, "timesteps_total": 0, "training_iteration": 0}
        tmp_callback = ParallelSNFSPBestResponseCallbacks()
        tmp_callback.on_train_result(trainer=br_trainers[0], result=final_train_result)
        print(f"\n\nexploitability: {final_train_result['z_avg_policy_exploitability']}\n\n")
    else:
        train_thread_executor = ThreadPoolExecutor(max_workers=2)

        stop_training = False
        final_train_result = None
        while not stop_training:

            # synchronize opponent best response policies
            for trainer in br_trainers:
                opponent_br_weights = trainer.other_trainer.get_weights(["best_response"])
                opponent_br_weights["opponent_best_response"] = opponent_br_weights["best_response"]
                del opponent_br_weights["best_response"]
                trainer.set_weights(opponent_br_weights)

            # do a train step for both BRs at the same time
            # train_result_futures = tuple(train_thread_executor.submit(lambda: trainer.train()) for trainer in br_trainers)
            # both_train_iter_results = tuple(future.result() for future in train_result_futures)

            both_train_iter_results = tuple(trainer.train() for trainer in br_trainers)

            if train_iter_count % checkpoint_every_n_iters == 0:
                checkpoint_brs_for_average_policy(checkpoint_count=checkpoint_count, both_train_iter_results=both_train_iter_results)
                checkpoint_count += 1

            train_iter_count += 1
            print("printing results..")

            if print_train_results:
                for br_trainer, train_iter_results in zip(br_trainers, both_train_iter_results):
                    print(f"trainer {br_trainer.player}")
                    # Delete verbose debugging info
                    if "hist_stats" in train_iter_results:
                        del train_iter_results["hist_stats"]
                    if "td_error" in train_iter_results["info"]["learner"]["best_response"]:
                        del train_iter_results["info"]["learner"]["best_response"]["td_error"]

                    if br_trainer.player == 1:
                        assert br_trainer.other_trainer.player == 0, br_trainer.other_trainer.player
                        br_trainer.other_trainer.other_player_latest_train_result = deepcopy(train_iter_results)
                        assert len(train_iter_results) > 0

                    if br_trainer.player == 0:
                        log(pretty_dict_str(train_iter_results))

                        # TODO have better stopping rule
                        if (train_iter_results["episodes_total"] >= int(500000)
                                and "z_avg_policy_exploitability" in train_iter_results
                                and train_iter_results["avg_br_reward_both_players"] < 0.03
                        ):
                            final_train_result = deepcopy(train_iter_results)
                            stop_training = True

                    assert br_trainer.player in [0,1]

                    log(f"Trainer {br_trainer.player} logdir is {br_trainer.logdir}")

    avg_policy_specs = []
    for player, trainer in enumerate(br_trainers):
        player_avg_policy = trainer.other_trainer.workers.local_worker().policy_map["opponent_average_policy"]

        avg_pol_spec_list: List[PayoffTableStrategySpec] = player_avg_policy.br_specs.to_list()
        json_spec_list = [spec.to_json() for spec in avg_pol_spec_list]

        avg_policy_spec = PayoffTableStrategySpec(
            strategy_id=f"avg_policy_player_{player}_{datetime_str()}",
            metadata={
                "cfp_pure_strat_specs": json_spec_list
            })
        avg_policy_specs.append(avg_policy_spec)

    assert final_train_result is not None
    return avg_policy_specs, final_train_result["z_avg_policy_exploitability"], final_train_result["episodes_total"], final_train_result["timesteps_total"]


def solve_restricted_game_cfp(log_dir: str, player_to_br_action_specs: Dict[int, List[PayoffTableStrategySpec]]) -> Tuple[List[PayoffTableStrategySpec], Dict]:
    logging.basicConfig(level=logging.INFO)
    results_dir = os.path.join(log_dir, f"cfp_{datetime_str()}")
    print(f"results dir is {results_dir}")

    def extra_action_out_fn(policy: Policy, input_dict, state_batches, model,
                            action_dist: ActionDistribution) -> Dict[str, TensorType]:
        action = action_dist.deterministic_sample()
        action_probs = torch.zeros_like(policy.q_values).long()
        action_probs[0][action[0]] = 1.0
        return {"q_values": policy.q_values, "action_probs": action_probs}

    policy_class = SimpleQTorchPolicy.with_updates(
        extra_action_out_fn=extra_action_out_fn
    )

    avg_policy_specs, exploitability, cfp_episodes, cfp_steps = train_poker_off_policy_rl_nfsp(
        print_train_results=True,
        br_trainer_class=DQNTrainer,
        br_policy_class=policy_class,
        hyperparams=kuhn_dqn_params,
        results_dir=results_dir,
        player_to_base_game_action_specs=player_to_br_action_specs)

    return avg_policy_specs, {"exploitability": exploitability,
                              "timesteps_total": cfp_steps,
                              "episodes_total": cfp_episodes
                              }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    results_dir = os.path.join(os.path.dirname(grl.__file__), "data")
    print(f"results dir is {results_dir}")


    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='dqn', help="[SAC|DQN]")
    args = parser.parse_args()

    if args.algo.lower() == 'dqn':

        def extra_action_out_fn(policy: Policy, input_dict, state_batches, model,
                                action_dist: ActionDistribution) -> Dict[str, TensorType]:
            action = action_dist.deterministic_sample()
            action_probs = torch.zeros_like(policy.q_values).long()
            action_probs[0][action[0]] = 1.0
            return {"q_values": policy.q_values, "action_probs": action_probs}

        policy_class = SimpleQTorchPolicy.with_updates(
            extra_action_out_fn=extra_action_out_fn
        )

        train_poker_off_policy_rl_nfsp(print_train_results=True,
                                       br_trainer_class=DQNTrainer,
                                       br_policy_class=policy_class,
                                       hyperparams=kuhn_dqn_params,
                                       results_dir=results_dir)
    elif args.algo.lower() == 'sac':

        def behaviour_logits_fetches(
                policy: Policy, input_dict: Dict[str, TensorType],
                state_batches: List[TensorType], model: ModelV2,
                action_dist: TorchDistributionWrapper) -> Dict[str, TensorType]:
            """Defines extra fetches per action computation.

            Args:
                policy (Policy): The Policy to perform the extra action fetch on.
                input_dict (Dict[str, TensorType]): The input dict used for the action
                    computing forward pass.
                state_batches (List[TensorType]): List of state tensors (empty for
                    non-RNNs).
                model (ModelV2): The Model object of the Policy.
                action_dist (TorchDistributionWrapper): The instantiated distribution
                    object, resulting from the model's outputs and the given
                    distribution class.

            Returns:
                Dict[str, TensorType]: Dict with extra tf fetches to perform per
                    action computation.
            """
            return {
                "behaviour_logits": policy.logits,
            }

        policy_class = SACTorchPolicy.with_updates(
            extra_action_out_fn=behaviour_logits_fetches
        )

        train_poker_off_policy_rl_nfsp(print_train_results=True,
                                       br_trainer_class=SACTrainer,
                                       br_policy_class=policy_class,
                                       hyperparams=kuhn_sac_params,
                                       results_dir=results_dir)
    else:
        raise NotImplementedError(f"Choice for arg 'algo': {args.algo} isn't implemented.")