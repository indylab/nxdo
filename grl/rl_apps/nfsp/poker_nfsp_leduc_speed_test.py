import os
import logging
import numpy as np
from typing import Any, Tuple, Callable, Type, Dict
import tempfile
import argparse

from gym.spaces import Space
import copy

import ray
from ray.rllib.utils import merge_dicts, try_import_torch
torch, _ = try_import_torch()

from ray.rllib.agents import Trainer
from ray.rllib.agents.dqn import DQNTrainer, SimpleQTorchPolicy
from ray.rllib.utils.typing import AgentID, PolicyID
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch
import grl
from grl.utils import pretty_dict_str, datetime_str, copy_attributes

from grl.rl_apps.kuhn_poker_p2sro.poker_multi_agent_env import PokerMultiAgentEnv

from grl.rl_apps.nfsp.config import leduc_dqn_params
from grl.nfsp_rllib.nfsp import NFSPTrainer, NFSPTorchAveragePolicy, get_store_to_avg_policy_buffer_fn
from grl.rl_apps.nfsp.openspiel_utils import nfsp_measure_exploitability_nonlstm
from grl.rllib_tools.valid_actions_fcnet import LeducDQNFullyConnectedNetwork
from grl.rllib_tools.space_saving_logger import SpaceSavingLogger

logger = logging.getLogger(__name__)



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
            return "z_avg_policy_exploitability" in result

        return SpaceSavingLogger(config=config, logdir=logdir, should_log_result_fn=_should_log)

    return trainer_logger_creator


def checkpoint_dir(trainer: Trainer):
    return os.path.join(trainer.logdir, "br_checkpoints")


def train_poker_off_policy_rl_nfsp(results_dir: str,
                                   br_trainer_class: Type[Trainer],
                                   br_policy_class: Type[Policy],
                                   get_br_config: Callable[[Space], Dict],
                                   print_train_results: bool = True):
    ray.init(address='auto', _redis_password='5241590000000000', local_mode=False)

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

    def assert_not_called(agent_id):
        assert False

    env_config = {
        'version': "leduc_poker",
        "append_valid_actions_mask_to_obs": True,
        'fixed_players': True,
    }
    tmp_env = PokerMultiAgentEnv(env_config=env_config)

    avg_trainer = NFSPTrainer(config={
        "log_level": "DEBUG",
        "framework": "torch",
        "env": PokerMultiAgentEnv,
        "env_config": env_config,
        "num_gpus": 0.1,
        "num_workers": 0,
        "num_gpus_per_worker": 0.0,
        "num_envs_per_worker": 1,
        "multiagent": {
            "policies_to_train": ["average_policy_0", "average_policy_1"],
            "policies": {
                "average_policy_0": (NFSPTorchAveragePolicy, tmp_env.observation_space, tmp_env.action_space, {
                    "model": {
                        "fcnet_activation": "relu",
                        "fcnet_hiddens": [128],
                        "custom_model": LeducDQNFullyConnectedNetwork,
                    }
                }),
                "average_policy_1": (NFSPTorchAveragePolicy, tmp_env.observation_space, tmp_env.action_space, {
                    "model": {
                        "fcnet_activation": "relu",
                        "fcnet_hiddens": [128],
                        "custom_model": LeducDQNFullyConnectedNetwork,
                    }
                }),
            },
            "policy_mapping_fn": assert_not_called,
        },

    }, logger_creator=get_trainer_logger_creator(base_dir=results_dir, env_class=PokerMultiAgentEnv))

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
            if training_iteration == 1 or training_iteration % 2000 == 0:
                local_avg_policy_0 = trainer.workers.local_worker().policy_map["average_policy_0"]
                local_avg_policy_1 = trainer.workers.local_worker().policy_map["average_policy_1"]
                exploitability = nfsp_measure_exploitability_nonlstm(
                    rllib_policies=[local_avg_policy_0, local_avg_policy_1],
                    poker_game_version="leduc_poker")
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
        "env": PokerMultiAgentEnv,
        "env_config": env_config,
        "gamma": 1.0,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        # "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_gpus": 0.0,
        "num_workers": 4,
        "num_gpus_per_worker": 0.0,
        "num_envs_per_worker": 8,
        "multiagent": {
            "policies_to_train": ["best_response_0", "best_response_1"],
            "policies": {
                "average_policy_0": (NFSPTorchAveragePolicy, tmp_env.observation_space, tmp_env.action_space, {
                    "model": {
                        "fcnet_activation": "relu",
                        "fcnet_hiddens": [128],
                        "custom_model": LeducDQNFullyConnectedNetwork,
                    },
                    "explore": False,
                }),
                "best_response_0": (br_policy_class, tmp_env.observation_space, tmp_env.action_space, {}),

                "average_policy_1": (NFSPTorchAveragePolicy, tmp_env.observation_space, tmp_env.action_space, {
                    "model": {
                        "fcnet_activation": "relu",
                        "fcnet_hiddens": [128],
                        "custom_model": LeducDQNFullyConnectedNetwork,
                    },
                    "explore": False,
                }),
                "best_response_1": (br_policy_class, tmp_env.observation_space, tmp_env.action_space, {}),
            },
            "policy_mapping_fn": select_policy,
        },
    }
    br_trainer_config = merge_dicts(br_trainer_config, get_br_config(tmp_env.action_space))

    br_trainer_config["rollout_fragment_length"] = br_trainer_config["rollout_fragment_length"] // max(1, br_trainer_config["num_workers"] * br_trainer_config["num_envs_per_worker"] )


    br_trainer = br_trainer_class(config=br_trainer_config, logger_creator=get_trainer_logger_creator(base_dir=results_dir,
                                                                                          env_class=PokerMultiAgentEnv))


    assert isinstance(br_trainer.workers.local_worker().policy_map["average_policy_1"].model, LeducDQNFullyConnectedNetwork)
    assert isinstance(br_trainer.workers.local_worker().policy_map["average_policy_0"].model, LeducDQNFullyConnectedNetwork)
    assert isinstance(avg_trainer.workers.local_worker().policy_map["average_policy_0"].model, LeducDQNFullyConnectedNetwork)
    assert isinstance(avg_trainer.workers.local_worker().policy_map["average_policy_0"].model, LeducDQNFullyConnectedNetwork)

    br_trainer.latest_avg_trainer_result = None
    train_iter_count = 0

    for trainer in [br_trainer, avg_trainer]:
        for policy_id, policy in trainer.workers.local_worker().policy_map.items():
            policy.policy_id = policy_id

    br_trainer.workers.local_worker().policy_map["average_policy_0"] = avg_trainer.workers.local_worker().policy_map["average_policy_0"]
    br_trainer.workers.local_worker().policy_map["average_policy_1"] = avg_trainer.workers.local_worker().policy_map["average_policy_1"]
    # br_trainer.set_weights(avg_trainer.get_weights(["average_policy_0"]))
    # br_trainer.set_weights(avg_trainer.get_weights(["average_policy_1"]))
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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    results_dir = os.path.join(os.path.dirname(grl.__file__), "data")
    print(f"results dir is {results_dir}")


    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='dqn', help="[SAC|DQN]")
    args = parser.parse_args()

    if args.algo.lower() == 'dqn':
        train_poker_off_policy_rl_nfsp(print_train_results=True,
                                       br_trainer_class=DQNTrainer,
                                       br_policy_class=SimpleQTorchPolicy,
                                       get_br_config=leduc_dqn_params,
                                       results_dir=results_dir)
    # elif args.algo.lower() == 'sac':
    #     train_off_policy_rl_nfsp(print_train_results=True,
    #                                    br_trainer_class=SACTrainer,
    #                                    br_policy_class=SACTorchPolicy,
    #                                    get_br_config=kuhn_sac_params,
    #                                    results_dir=results_dir)
    else:
        raise NotImplementedError(f"Choice for arg 'algo': {args.algo} isn't implemented.")