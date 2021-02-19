import os
import time
import logging
import tempfile

from typing import Dict, List, Type, Callable, Any
from tables.exceptions import HDF5ExtError
import deepdish
import argparse
import ray
from ray.rllib.utils import merge_dicts, try_import_torch

torch, _ = try_import_torch()
from ray.rllib.agents import Trainer
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.utils.typing import PolicyID
from ray.rllib.policy import Policy, TorchPolicy

from grl.utils.strategy_spec import StrategySpec
from grl.envs.poker_multi_agent_env import PokerMultiAgentEnv
from grl.utils.common import pretty_dict_str, datetime_str, ensure_dir
from grl.p2sro.p2sro_manager import RemoteP2SROManagerClient, P2SROManagerWithServer
from grl.p2sro.p2sro_manager.utils import get_latest_metanash_strategies, PolicySpecDistribution
from grl.rllib_tools.space_saving_logger import SpaceSavingLogger
from grl.rl_apps.scenarios.poker import scenarios
from grl.rl_apps.scenarios.stopping_conditions import StoppingCondition
from grl.rllib_tools.safe_convert_to_torch_tensor import safe_convert_to_torch_tensor
from grl.rl_apps.scenarios.ray_setup import init_ray_for_scenario

logger = logging.getLogger(__name__)


def save_policy_checkpoint(trainer: Trainer,
                           player: int,
                           save_dir: str,
                           policy_id_to_save: PolicyID,
                           checkpoint_name: str,
                           additional_data: Dict[str, Any]):
    date_time = datetime_str()
    checkpoint_name = f"policy_{checkpoint_name}_{date_time}.h5"
    checkpoint_path = os.path.join(save_dir, checkpoint_name)
    br_weights = trainer.get_weights([policy_id_to_save])[policy_id_to_save]
    br_weights = {k.replace(".", "_dot_"): v for k, v in
                  br_weights.items()}  # periods cause HDF5 NaturalNaming warnings
    ensure_dir(file_path=checkpoint_path)
    num_save_attempts = 5

    checkpoint_data = {
        "weights": br_weights,
        "player": player,
        "date_time_str": date_time,
        "seconds_since_epoch": time.time(),
    }
    checkpoint_data.update(additional_data)

    for attempt in range(num_save_attempts):
        try:
            deepdish.io.save(path=checkpoint_path, data=checkpoint_data)
            break
        except HDF5ExtError:
            if attempt + 1 == num_save_attempts:
                raise
            time.sleep(1.0)
    return checkpoint_path


def load_pure_strat(policy: Policy, pure_strat_spec: StrategySpec = None, checkpoint_path: str = None):
    if pure_strat_spec is not None and checkpoint_path is not None:
        raise ValueError("Can only pass pure_strat_spec or checkpoint_path but not both")
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
    def load_pure_strat_cached(policy: TorchPolicy, pure_strat_spec):

        pure_strat_checkpoint_path = pure_strat_spec.metadata["checkpoint_path"]

        if pure_strat_checkpoint_path in cache:
            weights = cache[pure_strat_checkpoint_path]
        else:
            checkpoint_data = deepdish.io.load(path=pure_strat_checkpoint_path)
            weights = checkpoint_data["weights"]
            weights = {k.replace("_dot_", "."): v for k, v in weights.items()}

            weights = safe_convert_to_torch_tensor(weights, device=policy.device)
            cache[pure_strat_checkpoint_path] = weights

        policy.model.load_state_dict(weights)
        policy.policy_spec = pure_strat_spec

    return load_pure_strat_cached
