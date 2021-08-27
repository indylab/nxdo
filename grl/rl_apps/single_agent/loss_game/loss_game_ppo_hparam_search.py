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
from ray.tune import choice
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.agents.ppo import PPOTrainer, PPOTorchPolicy
from grl.rllib_tools.space_saving_logger import get_trainer_logger_creator
from grl.utils.common import find_free_port
from grl.utils.common import data_dir
from grl.rllib_tools.stochastic_sampling_ignore_kwargs import StochasticSamplingIgnoreKwargs
from grl.algos.p2sro.payoff_table import PayoffTable
from grl.algos.p2sro.p2sro_manager.utils import PolicySpecDistribution, get_latest_metanash_strategies
from grl.envs.loss_game_alpha_multi_agent_env import LossGameAlphaMultiAgentEnv

from grl.rllib_tools.policy_checkpoints import load_pure_strat
from grl.rl_apps.scenarios.catalog import scenario_catalog
from grl.rl_apps.scenarios.psro_scenario import PSROScenario
from ray.tune.suggest.hyperopt import HyperOptSearch

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    tmp_br_env = LossGameAlphaMultiAgentEnv(env_config={
        "total_moves": 10,
        "alpha": 2.9,
    })
    br_obs_space = tmp_br_env.observation_space
    br_act_space = tmp_br_env.action_space

    experiment_name = f"loss_game_alpha_hparam_search"
    num_cpus = 90
    num_gpus = 0
    env_class = LossGameAlphaMultiAgentEnv

    br_player = 1

    env_config = {
        "total_moves": 10,
        "alpha": 2.9,
    }

    metanash_pol_scenario: PSROScenario = scenario_catalog.get(scenario_name="loss_game_psro_10_moves_alpha_2.9")

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
            return f"metanash"


    metanash_policy_model_config = metanash_pol_scenario.get_trainer_config(tmp_env)["model"]

    payoff_table_path = "/home/jblanier/git/grl/grl/data/loss_game_psro_10_moves_alpha_2.9/manager_12.17.29AM_May-18-2021/payoff_table_checkpoints/payoff_table_checkpoint_20.json"
    payoff_table = PayoffTable.from_json_file(json_file_path=payoff_table_path)
    latest_strategies: Dict[int, PolicySpecDistribution] = get_latest_metanash_strategies(
        payoff_table=payoff_table,
        as_player=br_player,
        as_policy_num=None,
        fictitious_play_iters=2000,
        mix_with_uniform_dist_coeff=0.0
    )
    opponent_policy_distribution: PolicySpecDistribution = latest_strategies[1 - br_player]


    class HyperParamSearchCallbacks(DefaultCallbacks):

        def on_episode_start(self, *, worker: "RolloutWorker", base_env: BaseEnv, policies: Dict[PolicyID, Policy],
                             episode: MultiAgentEpisode, env_index: int, **kwargs):
            super().on_episode_start(worker=worker, base_env=base_env, policies=policies, episode=episode,
                                     env_index=env_index, **kwargs)
            metanash_policy = worker.policy_map["metanash"]
            load_pure_strat(policy=metanash_policy, pure_strat_spec=opponent_policy_distribution.sample_policy_spec())

        def on_train_result(self, *, trainer, result: dict, **kwargs):
            super().on_train_result(trainer=trainer, result=result, **kwargs)
            result["br_reward_mean"] = result["policy_reward_mean"]["best_response"]


    hyperparams = {

        "exploration_config": {
            # The Exploration class to use. In the simplest case, this is the name
            # (str) of any class present in the `rllib.utils.exploration` package.
            # You can also provide the python class directly or the full location
            # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
            # EpsilonGreedy").
            "type": StochasticSamplingIgnoreKwargs,
            # Add constructor kwargs here (if any).
        },

        "framework": "torch",
        "callbacks": HyperParamSearchCallbacks,
        "env": env_class,
        "env_config": env_config,
        "gamma": 1.0,
        "multiagent": {
            "policies_to_train": ["best_response"],
            "policies": {
                "metanash": (
                    metanash_pol_scenario.policy_classes["metanash"], tmp_env.observation_space, tmp_env.action_space,
                    {
                        "model": metanash_policy_model_config,
                    }),
                "best_response": (PPOTorchPolicy, br_obs_space, br_act_space, {
                    "model": merge_dicts(MODEL_DEFAULTS, {
                        "fcnet_hiddens": [32, 32],
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
        "simple_optimizer": False,

        "batch_mode": "truncate_episodes",

        # Coefficient of the entropy regularizer.
        "entropy_coeff": choice([0.0, 0.1, 0.01, 0.001]),
        # Decay schedule for the entropy regularizer.
        "entropy_coeff_schedule": None,

        # Initial coefficient for KL divergence.
        "kl_coeff": 0.2,

        "lambda": choice([1.0, 0.9]),

        # Size of batches collected from each worker.
        "rollout_fragment_length": 256,
        # Number of timesteps collected for each SGD round. This defines the size
        # of each SGD epoch.
        "train_batch_size": choice([2048, 4096]),
        # Total SGD batch size across all devices for SGD. This defines the
        # minibatch size within each epoch.
        "sgd_minibatch_size": choice([128, 256, 512, 1024]),
        # Number of SGD iterations in each outer loop (i.e., number of epochs to
        # execute per train batch).
        "num_sgd_iter": choice([1, 5, 10, 30, 60]),
        # Stepsize of SGD.
        "lr": choice([5e-2, 5e-3, 5e-4, 5e-5, 5e-6]),
        # PPO clip parameter.
        "clip_param": choice([0.1, 0.2, 0.3]),
        # Clip param for the value function. Note that this is sensitive to the
        # scale of the rewards. If your expected V is large, increase this.
        "vf_clip_param": 10.0,
        # If specified, clip the global norm of gradients by this amount.
        "grad_clip": None,
        # Target value for KL divergence.
        "kl_target": choice([0.001, 0.01, 0.1]),
    }

    search = HyperOptSearch(metric="br_reward_mean", mode="max", n_initial_points=50)

    tune.run(run_or_experiment=trainer_class,
             name=experiment_name,
             metric="br_reward_mean",

             config=hyperparams,
             num_samples=200,
             search_alg=search,
             mode="max",
             local_dir=data_dir(),
             stop={"timesteps_total": int(400e3)},
             loggers=[get_trainer_logger_creator(
                 base_dir=data_dir(),
                 scenario_name=experiment_name,
                 should_log_result_fn=lambda result: result["training_iteration"] % 20 == 0)],
             )
