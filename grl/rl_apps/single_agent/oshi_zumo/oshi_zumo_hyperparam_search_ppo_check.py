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
# from ray.tune import choice
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.agents.ppo import PPOTrainer, PPOTorchPolicy
from grl.utils.strategy_spec import StrategySpec
from grl.rllib_tools.space_saving_logger import get_trainer_logger_creator
from grl.utils.common import find_free_port
from grl.utils.common import data_dir
from grl.envs.oshi_zumo_multi_agent_env import ThousandActionOshiZumoMultiAgentEnv
from grl.rllib_tools.policy_checkpoints import load_pure_strat
from grl.rl_apps.scenarios.catalog import scenario_catalog
from grl.rl_apps.scenarios.nfsp_scenario import NFSPScenario

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    tmp_br_env = ThousandActionOshiZumoMultiAgentEnv(env_config={
        'version': "oshi_zumo",
        'fixed_players': True,
        'append_valid_actions_mask_to_obs': False,
        'continuous_action_space': True,
    })
    br_obs_space = tmp_br_env.observation_space
    br_act_space = tmp_br_env.action_space

    experiment_name = f"oshi_zumo_hyper_param_search_ppo_cont_check"
    num_cpus = 12
    num_gpus = 0
    env_class = ThousandActionOshiZumoMultiAgentEnv

    br_player = 1
    avg_policy_player = 1 - br_player

    env_config = {
        'version': "oshi_zumo",
        'fixed_players': True,
        'append_valid_actions_mask_to_obs': True,
        'continuous_action_space': False,
        'individual_players_with_continuous_action_space': [br_player],
        'individual_players_with_orig_obs_space': [br_player],
    }

    avg_pol_scenario: NFSPScenario = scenario_catalog.get(scenario_name="1000_oshi_zumo_nfsp_larger_dqn_larger")

    trainer_class = PPOTrainer

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
        "/home/jblanier/git/grl/grl/data/1000_oshi_zumo_nfsp_larger_dqn_larger_sparse_10.56.11PM_Mar-24-20217lav0isx/avg_policy_checkpoint_specs/average_policy_player_0_iter_53000.json")


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
                "best_response": (PPOTorchPolicy, br_obs_space, br_act_space, {
                    "model": merge_dicts(MODEL_DEFAULTS, {
                        "fcnet_activation": "relu",
                        "fcnet_hiddens": [128, 128],
                        "custom_model": None,
                        "custom_action_dist": "TorchGaussianSquashedGaussian",
                    }),
                }),
            },
            "policy_mapping_fn": select_policy,
        },
        # Share layers for value function. If you set this to True, it's important
        # to tune vf_loss_coeff.
        "vf_share_layers": False,
        "num_gpus": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_workers": 4,
        "num_gpus_per_worker": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_envs_per_worker": 1,
        "metrics_smoothing_episodes": 5000,

        # Coefficient of the entropy regularizer.
        "entropy_coeff": 0.0,
        # Decay schedule for the entropy regularizer.
        "entropy_coeff_schedule": None,

        # "model": choice([
        #     merge_dicts(MODEL_DEFAULTS, {
        #         "fcnet_activation": "relu",
        #         "fcnet_hiddens": [128],
        #         "custom_model": None,
        #         "custom_action_dist": "TorchGaussianSquashedGaussian",
        #     }),
        #     merge_dicts(MODEL_DEFAULTS, {
        #         "fcnet_activation": "relu",
        #         "fcnet_hiddens": [128, 128],
        #         "custom_model": None,
        #         "custom_action_dist": "TorchGaussianSquashedGaussian",
        #     }),
        #     merge_dicts(MODEL_DEFAULTS, {
        #         "fcnet_activation": "relu",
        #         "fcnet_hiddens": [128, 128, 128],
        #         "custom_model": None,
        #         "custom_action_dist": "TorchGaussianSquashedGaussian",
        #     }),
        # ]),
        # Initial coefficient for KL divergence.
        "kl_coeff": 0.2,
        # Size of batches collected from each worker.
        "rollout_fragment_length": 256,
        # Number of timesteps collected for each SGD round. This defines the size
        # of each SGD epoch.
        "train_batch_size": 2048,
        # Total SGD batch size across all devices for SGD. This defines the
        # minibatch size within each epoch.
        "sgd_minibatch_size": 512,
        # Number of SGD iterations in each outer loop (i.e., number of epochs to
        # execute per train batch).
        "num_sgd_iter": 1,
        # Stepsize of SGD.
        "lr": 0.0005,
        # PPO clip parameter.
        "clip_param": 0.03,
        # Clip param for the value function. Note that this is sensitive to the
        # scale of the rewards. If your expected V is large, increase this.
        "vf_clip_param": 10.0,
        # If specified, clip the global norm of gradients by this amount.
        "grad_clip": None,
        # Target value for KL divergence.
        "kl_target": 0.001,
    }

    tune.run(run_or_experiment=trainer_class,
             name=experiment_name,
             metric="br_reward_mean",

             config=hyperparams,
             num_samples=2,
             search_alg=None,
             mode="max",
             local_dir=data_dir(),
             stop={"timesteps_total": int(3e6)},
             loggers=[get_trainer_logger_creator(
                 base_dir=data_dir(),
                 scenario_name=experiment_name,
                 should_log_result_fn=lambda result: result["training_iteration"] % 20 == 0)],
             )
