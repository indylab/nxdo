import os
import time
import logging
import numpy as np
from typing import Dict, List
import tempfile

import deepdish

import ray
from ray.rllib.agents.sac import SACTrainer, SACTorchPolicy
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
from ray.tune.logger import Logger, UnifiedLogger

import grl
from grl.p2sro.payoff_table import PayoffTableStrategySpec
from grl.utils import pretty_dict_str, datetime_str, ensure_dir
from grl.p2sro.p2sro_manager import RemoteP2SROManagerClient
from grl.p2sro.p2sro_manager.utils import get_latest_metanash_strategies, PolicySpecDistribution

from grl.rl_examples.particle_games.simple_push_multi_agent_env import SimplePushMultiAgentEnv
from grl.rl_examples.particle_games.simple_push_hparam_set import simple_push_sac_hparam_search, trial_name_creator

logger = logging.getLogger(__name__)

torch, _ = try_import_torch()


def checkpoint_dir(trainer: SACTrainer):
    return os.path.join(trainer.logdir, "br_checkpoints")


def load_pure_strat(policy: SACTorchPolicy, pure_strat_checkpoint_path: str):
    checkpoint_data = deepdish.io.load(path=pure_strat_checkpoint_path)
    weights = checkpoint_data["weights"]
    weights = {k.replace("_dot_", "."): v for k, v in weights.items()}
    policy.set_weights(weights=weights)


class SetFixedOpponentWeightsCallbacks(DefaultCallbacks):

    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        if not hasattr(worker, "weights_set"):
            fixed_opponent_policy: SACTorchPolicy = worker.policy_map["fixed_opponent"]
            load_pure_strat(policy=fixed_opponent_policy, pure_strat_checkpoint_path="/home/jblanier/git/grl/grl/data/SymmetricPushMultiAgentEnv_08.58.20PM_Dec-14-20202hc7iw01/br_checkpoints/policy_3_08.58.46PM_Dec-14-2020.h5")
            worker.weights_set = True


def search_sac_hyperparams():

    def select_policy(agent_id):
        if agent_id == 1:
            return "best_response"
        else:
            return "fixed_opponent"

    env_config = {}

    tmp_env = SimplePushMultiAgentEnv(env_config=env_config)

    trainer_config = {
        "log_level": "DEBUG",
        "callbacks": SetFixedOpponentWeightsCallbacks,
        "env": SimplePushMultiAgentEnv,
        "env_config": env_config,
        "gamma": 0.95,

        "num_gpus": 0.1,
        "num_workers": 19,
        "num_gpus_per_worker": 0.1,
        "num_envs_per_worker": 2,

        "multiagent": {
            "policies_to_train": ["best_response"],
            "policies": {
                "fixed_opponent": (SACTorchPolicy, tmp_env.observation_space, tmp_env.action_space, {
                    "Q_model": {
                        "fcnet_activation": "relu",
                        "fcnet_hiddens": [128, 128],
                    },
                    # Model options for the policy function.
                    "policy_model": {
                        "fcnet_activation": "relu",
                        "fcnet_hiddens": [128, 128],
                    },
                    "normalize_actions": False,
                }),
                "best_response": (SACTorchPolicy, tmp_env.observation_space, tmp_env.action_space, {}),
            },
            "policy_mapping_fn": select_policy,
        },
    }
    trainer_config = merge_dicts(trainer_config, simple_push_sac_hparam_search(action_space=tmp_env.action_space))

    analysis = ray.tune.run(
        name=f"SACHyperParamsSimplePushSymmetric_{datetime_str()}",
        with_server=False,
        server_port=7865,
        checkpoint_at_end=False,
        keep_checkpoints_num=0,
        num_samples=10,
        max_failures=0,
        reuse_actors=False,
        trial_name_creator=trial_name_creator,
        stop={
            "training_iteration": 600,
        },
        run_or_experiment=SACTrainer,
        config=trainer_config)
    return analysis


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    search_sac_hyperparams()

