import os
import time
import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Callable, Type, Dict, Union
import tempfile
import argparse
import random
from collections import defaultdict
from gym.spaces import Space

import deepdish

import ray
from ray.rllib import SampleBatch, Policy
from ray.rllib.agents import Trainer
from ray.rllib.agents.sac import SACTrainer, SACTorchPolicy
from ray.rllib.agents.dqn import DQNTrainer, DQNTorchPolicy, SimpleQTorchPolicy, SimpleQTFPolicy
from ray.rllib.utils import merge_dicts, try_import_torch
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

from grl.rl_examples.kuhn_poker_p2sro.poker_multi_agent_env import PokerMultiAgentEnv

from grl.p2sro.payoff_table import PayoffTableStrategySpec
from grl.rl_examples.nfsp.config import kuhn_dqn_params, kuhn_sac_params
from grl.rl_examples.kuhn_poker_p2sro.poker_utils import measure_exploitability_nonlstm, openspiel_policy_from_nonlstm_rllib_policy
from grl.nfsp_rllib.nfsp import NFSPTrainer, NFSPTorchAveragePolicy, get_store_to_avg_policy_buffer_fn
from grl.rl_examples.nfsp.openspiel_utils import snfsp_measure_exploitability_nonlstm


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




logger = logging.getLogger(__name__)

torch, _ = try_import_torch()


def save_snfsp_best_response_checkpoint(trainer: Trainer,
                                        policy_id_to_save: str,
                                      save_dir: str,
                                      timesteps_training_br: int,
                                      episodes_training_br: int):
    policy_name = policy_id_to_save
    date_time = datetime_str()
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


def create_metadata_with_new_checkpoint_for_current_best_response(trainer: Trainer,
                                                                policy_id_to_save: str,
                                                              save_dir: str,
                                                              timesteps_training_br: int,
                                                              episodes_training_br: int,
                                                              ):
    return {
        "checkpoint_path": save_snfsp_best_response_checkpoint(trainer=trainer,
                                                               policy_id_to_save=policy_id_to_save,
                                                               save_dir=save_dir,
                                                         timesteps_training_br=timesteps_training_br,
                                                         episodes_training_br=episodes_training_br),
        "timesteps_training_br": timesteps_training_br,
        "episodes_training_br": episodes_training_br
    }

def add_best_response_strat_to_average_policy(trainer: Trainer, br_spec: PayoffTableStrategySpec, reservoir_buffer_idx: int, avg_policy_id):
    def worker_add_spec(worker: RolloutWorker):
        avg_policy: Policy = worker.policy_map[avg_policy_id]
        if not hasattr(avg_policy, "br_specs"):
            avg_policy.br_specs = ReservoirBuffer()
        avg_policy.br_specs.add(br_spec, idx=reservoir_buffer_idx)
    trainer.workers.foreach_worker(worker_add_spec)


def get_trainer_logger_creator(base_dir: str, env_class):
    logdir_prefix = f"{env_class.__name__}_{datetime_str()}"

    def trainer_logger_creator(config):
        """Creates a Unified logger with a default logdir prefix
        containing the agent name and the env id
        """
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        logdir = tempfile.mkdtemp(
            prefix=logdir_prefix, dir=base_dir)
        return UnifiedLogger(config, logdir)

    return trainer_logger_creator


def checkpoint_dir(trainer: Trainer):
    return os.path.join(trainer.logdir, "br_checkpoints")


def train_poker_off_policy_rl_nfsp(results_dir: str,
                                   br_trainer_class: Type[Trainer],
                                   br_policy_class: Type[Policy],
                                   get_br_config: Callable[[Space], Dict],
                                   print_train_results: bool = True):
    ray.init(local_mode=True)

    def log(message, level=logging.INFO):
        logger.log(level, message)

    def select_policy(agent_id):
        random_sample = np.random.random()
        if agent_id == 0:
            if random_sample < 0.1:
                return "best_response_0"
            return "average_policy_0"
        elif agent_id == 1:
            if random_sample < 0.1:
                return "best_response_1"
            return "average_policy_1"
        else:
            raise ValueError(f"unexpected agent_id: {agent_id}")

    env_config = {
        'version': "kuhn_poker",
        'fixed_players': True,
    }
    tmp_env = PokerMultiAgentEnv(env_config=env_config)

    class SNFSPBestResponseCallbacks(DefaultCallbacks):

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
                    del postprocessed_batch.data["q_values"]
                    del postprocessed_batch.data["action_dist_inputs"]

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
                    del postprocessed_batch.data["action_probs"]

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
                # for policy_id, policy_samples in samples.policy_batches.items():
                #     if policy_id == br_policy_id:
                #         store_to_avg_policy_buffer(MultiAgentBatch(policy_batches={
                #             average_policy_id: policy_samples
                #         }, env_steps=policy_samples.count))
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
            if training_iteration == 1 or training_iteration % 1000 == 0:
                local_avg_policy_0 = trainer.workers.local_worker().policy_map["average_policy_0"]
                local_avg_policy_1 = trainer.workers.local_worker().policy_map["average_policy_1"]

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
            for average_policy_id in ["average_policy_0", "average_policy_1"]:
                avg_policy: Policy = worker.policy_map[average_policy_id]
                if not hasattr(avg_policy, "br_specs"):
                    return
                new_br_spec = avg_policy.br_specs.sample()[0]
                if hasattr(avg_policy, "policy_spec") and new_br_spec == avg_policy.policy_spec:
                    return
                load_pure_strat(policy=avg_policy, pure_strat_spec=new_br_spec)

    br_trainer_config = {
        "log_level": "DEBUG",
        "callbacks": SNFSPBestResponseCallbacks,
        "env": PokerMultiAgentEnv,
        "env_config": env_config,
        "gamma": 1.0,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        # "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_gpus": 0,
        "num_workers": 0,
        "num_gpus_per_worker": 0,
        "num_envs_per_worker": 1,
        "multiagent": {
            "policies_to_train": ["best_response_0", "best_response_1"],
            "policies": {
                "average_policy_0": (br_policy_class, tmp_env.observation_space, tmp_env.action_space, {}),
                "best_response_0": (br_policy_class, tmp_env.observation_space, tmp_env.action_space, {}),
                "average_policy_1": (br_policy_class, tmp_env.observation_space, tmp_env.action_space, {}),
                "best_response_1": (br_policy_class, tmp_env.observation_space, tmp_env.action_space, {}),
            },
            "policy_mapping_fn": select_policy,
        },
    }
    br_trainer_config = merge_dicts(br_trainer_config, get_br_config(tmp_env.action_space))

    br_trainer = br_trainer_class(config=br_trainer_config, logger_creator=get_trainer_logger_creator(base_dir=results_dir,
                                                                                          env_class=PokerMultiAgentEnv))

    def checkpoint_br_for_average_policy(checkpoint_count, train_iter_results=None):
        log("\nCHECKPOINTING BR\n")
        if train_iter_results is not None:
            total_timesteps_training_br = train_iter_results["timesteps_total"]
            total_episodes_training_br = train_iter_results["episodes_total"]
        else:
            total_timesteps_training_br = 0
            total_episodes_training_br = 0

        for average_policy_id, br_policy_id in [("average_policy_0", "best_response_0"), ("average_policy_1", "best_response_1")]:

            local_avg_policy = br_trainer.workers.local_worker().policy_map[average_policy_id]
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
                    policy_id_to_save=br_policy_id,
                    trainer=br_trainer,
                    save_dir=checkpoint_dir(trainer=br_trainer),
                    timesteps_training_br=total_timesteps_training_br,
                    episodes_training_br=total_episodes_training_br,
                )
                log(checkpoint_metadata)

                br_checkpoint_spec = PayoffTableStrategySpec(strategy_id=str(checkpoint_count),
                                                             metadata=checkpoint_metadata)
                add_best_response_strat_to_average_policy(trainer=br_trainer, br_spec=br_checkpoint_spec,
                                                          reservoir_buffer_idx=add_idx,
                                                          avg_policy_id=average_policy_id)

    train_iter_count = 0

    checkpoint_every_n_iters = 20
    checkpoint_count = 0
    print("starting")
    checkpoint_br_for_average_policy(checkpoint_count=checkpoint_count, train_iter_results=None)
    while True:
        train_iter_results = br_trainer.train()  # do a step (or several) in the main RL loop

        if train_iter_count % checkpoint_every_n_iters == 0:
            checkpoint_br_for_average_policy(checkpoint_count=checkpoint_count, train_iter_results=train_iter_results)
            checkpoint_count += 1

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
                                       get_br_config=kuhn_dqn_params,
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
                                       get_br_config=kuhn_sac_params,
                                       results_dir=results_dir)
    else:
        raise NotImplementedError(f"Choice for arg 'algo': {args.algo} isn't implemented.")