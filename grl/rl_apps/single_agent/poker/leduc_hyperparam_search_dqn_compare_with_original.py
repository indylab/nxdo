import logging
import os
from typing import Dict

import ray
from ray.rllib import BaseEnv
from ray.rllib.utils import try_import_torch

torch, _ = try_import_torch()

from ray.rllib.utils import merge_dicts
from ray.rllib.utils.typing import PolicyID
from ray.rllib.models import MODEL_DEFAULTS
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.agents.dqn import DQNTrainer
from grl.rllib_tools.modified_policies.simple_q_torch_policy import SimpleQTorchPolicyPatched
from grl.utils.strategy_spec import StrategySpec
from grl.rllib_tools.space_saving_logger import get_trainer_logger_creator
from grl.utils.common import find_free_port
from grl.utils.common import data_dir
from grl.envs.poker_multi_agent_env import PokerMultiAgentEnv
from grl.rllib_tools.policy_checkpoints import load_pure_strat
from grl.rl_apps.scenarios.catalog import scenario_catalog
from grl.rl_apps.scenarios.nfsp_scenario import NFSPScenario
from grl.rllib_tools.models.valid_actions_fcnet import get_valid_action_fcn_class_for_env

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    experiment_name = f"leduc_hyper_param_search_dqn"
    num_cpus = 32
    num_gpus = 0
    env_class = PokerMultiAgentEnv

    br_player = 1
    avg_policy_player = 1 - br_player

    env_config = {
        "version": "leduc_poker",
        "fixed_players": True,
        "append_valid_actions_mask_to_obs": True,
    }

    avg_pol_scenario: NFSPScenario = scenario_catalog.get(scenario_name="leduc_nfsp_dqn")

    trainer_class = DQNTrainer

    tmp_env = env_class(env_config=env_config)

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


    def select_policy(agent_id):
        if agent_id == br_player:
            return "best_response"
        else:
            return f"average_policy"


    avg_policy_model_config = avg_pol_scenario.get_avg_trainer_config(tmp_env)["model"]

    player_0_avg_pol_spec = StrategySpec.from_json_file(
        "/home/jblanier/git/grl/grl/data/leduc_nfsp_dqn_sparse_02.34.06PM_Apr-08-2021bt5ym0l8/avg_policy_checkpoint_specs/average_policy_player_0_iter_263000.json"
    )


    class HyperParamSearchCallbacks(DefaultCallbacks):

        def on_episode_start(self, *, worker: "RolloutWorker", base_env: BaseEnv, policies: Dict[PolicyID, Policy],
                             episode: MultiAgentEpisode, env_index: int, **kwargs):
            super().on_episode_start(worker=worker, base_env=base_env, policies=policies, episode=episode,
                                     env_index=env_index, **kwargs)
            if not hasattr(worker, "avg_pol_loaded") or not worker.avg_pol_loaded:
                avg_policy = worker.policy_map["average_policy"]
                load_pure_strat(policy=avg_policy, pure_strat_spec=player_0_avg_pol_spec)
                worker.avg_pol_loaded = True

        def on_train_result(self, *, trainer, result: dict, **kwargs):
            super().on_train_result(trainer=trainer, result=result, **kwargs)
            result["br_reward_mean"] = result["policy_reward_mean"]["best_response"]


    hyperparams = {
        "framework": "torch",
        "callbacks": HyperParamSearchCallbacks,
        "env": env_class,
        "env_config": env_config,
        "gamma": 1.0,
        "multiagent": {
            "policies_to_train": ["best_response"],
            "policies": {
                "average_policy": (
                    avg_pol_scenario.policy_classes["average_policy"], tmp_env.observation_space, tmp_env.action_space,
                    {
                        "model": avg_policy_model_config,
                        "explore": False,
                    }),
                "best_response": (SimpleQTorchPolicyPatched, tmp_env.observation_space, tmp_env.action_space, {
                    "model": merge_dicts(MODEL_DEFAULTS, {
                        "fcnet_activation": "relu",
                        "fcnet_hiddens": [128],
                        "custom_model": get_valid_action_fcn_class_for_env(env=tmp_env),
                    }),
                }),
            },
            "policy_mapping_fn": select_policy,
        },

        "num_gpus": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_workers": 4,
        "num_envs_per_worker": 32,
        "num_gpus_per_worker": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "metrics_smoothing_episodes": 5000,
    }
    hyperparams = merge_dicts(hyperparams, avg_pol_scenario.get_trainer_config(tmp_env))
    hyperparams["metrics_smoothing_episodes"] = 5000

    # search = HyperOptSearch(metric="br_reward_mean", mode="max", n_initial_points=20)

    tune.run(run_or_experiment=trainer_class,
             name=experiment_name,
             metric="br_reward_mean",
             trial_name_creator=lambda _: "original_nxdo_paper",
             config=hyperparams,
             num_samples=3,
             # search_alg=search,
             mode="max",
             local_dir=data_dir(),
             stop={"timesteps_total": int(3e6)},
             loggers=[get_trainer_logger_creator(
                 base_dir=data_dir(),
                 scenario_name=experiment_name,
                 should_log_result_fn=lambda result: result["training_iteration"] % 20 == 0)],
             )
