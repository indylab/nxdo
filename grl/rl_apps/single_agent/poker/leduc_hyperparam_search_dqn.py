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
from ray.tune import choice, loguniform
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
from ray.tune.suggest.hyperopt import HyperOptSearch
from grl.rllib_tools.valid_actions_epsilon_greedy import ValidActionsEpsilonGreedy
from grl.rllib_tools.models.valid_actions_fcnet import get_valid_action_fcn_class_for_env

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    experiment_name = f"leduc_hyper_param_search_dqn"
    num_cpus = 60
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

        # Number of atoms for representing the distribution of return. When
        # this is greater than 1, distributional Q-learning is used.
        # the discrete supports are bounded by v_min and v_max
        "num_atoms": 1,
        "v_min": -10.0,
        "v_max": 10.0,
        # Whether to use noisy network
        "noisy": False,
        # control the initial value of noisy nets
        "sigma0": 0.5,
        # Whether to use dueling dqn
        "dueling": False,
        # Dense-layer setup for each the advantage branch and the value branch
        # in a dueling architecture.
        "hiddens": [256],
        # Whether to use double dqn
        "double_q": True,

        # N-step Q learning
        "n_step": 1,

        # === Exploration Settings ===
        "exploration_config":
            {
                # The Exploration class to use.
                "type": ValidActionsEpsilonGreedy,
                # Config for the Exploration class' constructor:
                "initial_epsilon": 0.06,
                "final_epsilon": 0.001,
                "epsilon_timesteps": choice([int(20e6), int(2e6), int(2e5)])  # Timesteps over which to anneal epsilon.
            },

        # Switch to greedy actions in evaluation workers.
        "evaluation_config": {
            "explore": False,
        },
        "explore": True,

        # Update the target network every `target_network_update_freq` steps.
        "target_network_update_freq": choice([10000, 1000, 100000]),

        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": choice([int(2e5), int(1e5), int(5e4)]),

        # If True prioritized replay buffer will be used.
        "prioritized_replay": False,
        # Alpha parameter for prioritized replay buffer.
        "prioritized_replay_alpha": 0.0,
        # Beta parameter for sampling from prioritized replay buffer.
        "prioritized_replay_beta": 0.0,
        # Final value of beta (by default, we use constant beta=0.4).
        "final_prioritized_replay_beta": 0.0,
        # Time steps over which the beta parameter is annealed.
        "prioritized_replay_beta_annealing_timesteps": 20000,
        # Epsilon to add to the TD errors when updating priorities.
        "prioritized_replay_eps": 0.0,
        # Whether to LZ4 compress observations
        "compress_observations": True,
        # Callback to run before learning on a multi-agent batch of experiences.
        # "before_learn_on_batch": debug_before_learn_on_batch,
        # If set, this will fix the ratio of replayed from a buffer and learned on
        # timesteps to sampled from an environment and stored in the replay buffer
        # timesteps. Otherwise, the replay will proceed at the native ratio
        # determined by (train_batch_size / rollout_fragment_length).
        "training_intensity": None,

        # === Optimization ===
        # Learning rate for adam optimizer
        "lr": loguniform(0.0001, 0.1),
        # Learning rate schedule
        "lr_schedule": None,
        # Adam epsilon hyper parameter
        "adam_epsilon": choice([1e-8, 1e-6, 1e-4, 1e-2]),
        # If not None, clip gradients during optimization at this value
        "grad_clip": None,
        # How many steps of the model to sample before learning starts.
        "learning_starts": 16000,
        # Update the replay buffer with this many samples at once. Note that
        # this setting applies per-worker if num_workers > 1.q
        "rollout_fragment_length": choice([4, 8, 16, 32]),
        "batch_mode": "truncate_episodes",

        # Size of a batch sampled from replay buffer for training. Note that
        # if async_updates is set, then each worker returns gradients for a
        # batch of this size.
        "train_batch_size": choice([4096, 2048, 1024]),

        # Whether to compute priorities on workers.
        "worker_side_prioritization": False,
        # Prevent iterations from going lower than this time span
        "min_iter_time_s": 0,
        # Minimum env steps to optimize for per train call. This value does
        # not affect learning (JB: this is a lie!), only the length of train iterations.
        "timesteps_per_iteration": 0,
    }

    search = HyperOptSearch(metric="br_reward_mean", mode="max", n_initial_points=20)

    tune.run(run_or_experiment=trainer_class,
             name=experiment_name,
             metric="br_reward_mean",

             config=hyperparams,
             num_samples=200000000,
             search_alg=search,
             mode="max",
             local_dir=data_dir(),
             stop={"timesteps_total": int(3e6)},
             loggers=[get_trainer_logger_creator(
                 base_dir=data_dir(),
                 scenario_name=experiment_name,
                 should_log_result_fn=lambda result: result["training_iteration"] % 20 == 0)],
             )
