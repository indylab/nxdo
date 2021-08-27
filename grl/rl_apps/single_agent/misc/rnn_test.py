import logging
import os

import ray
from ray.rllib.utils import merge_dicts, try_import_torch

torch, _ = try_import_torch()

from ray.rllib.agents.ppo import PPOTrainer, PPOTorchPolicy
from grl.utils.common import pretty_dict_str

from grl.rllib_tools.space_saving_logger import get_trainer_logger_creator
from grl.utils.common import find_free_port
from grl.envs.rnn_test_env import RNNTestMultiAgentEnv
from grl.utils.common import data_dir, datetime_str
from grl.rl_apps.scenarios.trainer_configs.mutigrid_psro_configs import psro_multigrid_ppo_params

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    num_cpus = 16
    num_gpus = 2
    env_class = RNNTestMultiAgentEnv
    trainer_class = PPOTrainer
    policy_classes = {
        "best_response": PPOTorchPolicy
    }
    get_trainer_config = psro_multigrid_ppo_params

    address_info = ray.init(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        object_store_memory=int(1073741824 * 1),
        local_mode=False,
        include_dashboard=True,
        dashboard_host="0.0.0.0",
        dashboard_port=find_free_port(),
        ignore_reinit_error=True,
        logging_level=logging.INFO,
        log_to_driver=os.getenv("RAY_LOG_TO_DRIVER", False))

    tmp_env = RNNTestMultiAgentEnv()

    trainer_config = {
        "env": env_class,
        "env_config": {},
        "gamma": 1.0,
        "num_gpus": 0,
        "num_workers": 0,
        "num_envs_per_worker": 1,
        "multiagent": {
            "policies_to_train": [f"best_response"],
            "policies": {
                f"best_response": (
                    policy_classes["best_response"], tmp_env.observation_space, tmp_env.action_space, {}),
            },
            "policy_mapping_fn": lambda agent_id: "best_response",
        },
    }
    trainer_config = merge_dicts(trainer_config, get_trainer_config(tmp_env))
    trainer = trainer_class(config=trainer_config,
                            logger_creator=get_trainer_logger_creator(
                                base_dir=data_dir(),
                                scenario_name=f"rnn_test_{datetime_str()}",
                                should_log_result_fn=lambda result: result["training_iteration"] % 20 == 0))

    while True:
        train_iter_results = trainer.train()  # do a step (or several) in the main RL loop

        # Delete verbose debugging info before printing
        if "hist_stats" in train_iter_results:
            del train_iter_results["hist_stats"]
        if "td_error" in train_iter_results["info"]["learner"][f"best_response"]:
            del train_iter_results["info"]["learner"][f"best_response"]["td_error"]

        print(pretty_dict_str(train_iter_results))
