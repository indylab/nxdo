import logging
import os
from typing import Dict

import ray
from ray.rllib import BaseEnv
from ray.rllib.utils import try_import_torch

torch, _ = try_import_torch()

from ray.rllib.utils.typing import PolicyID
from ray import tune
from ray.tune import choice
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from grl.utils.strategy_spec import StrategySpec
from grl.rllib_tools.space_saving_logger import get_trainer_logger_creator
from grl.utils.common import find_free_port
from grl.utils.common import data_dir
from grl.envs.oshi_zumo_multi_agent_env import ThousandActionOshiZumoMultiAgentEnv
from grl.rllib_tools.policy_checkpoints import load_pure_strat
from grl.rl_apps.scenarios.catalog import scenario_catalog
from grl.rl_apps.scenarios.nfsp_scenario import NFSPScenario
from ray.tune.suggest.hyperopt import HyperOptSearch
from grl.rllib_tools.modified_policies.sac_torch_policy_squashed import SACTorchPolicySquashed
from ray.rllib.agents.sac.sac import SACTrainer

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

    experiment_name = f"oshi_zumo_hyper_param_search_sac_cont"
    num_cpus = 80
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

    trainer_class = SACTrainer

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
                "best_response": (SACTorchPolicySquashed, br_obs_space, br_act_space, {}),
            },
            "policy_mapping_fn": select_policy,
        },
        "num_gpus": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_workers": 4,
        "num_gpus_per_worker": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_envs_per_worker": 1,
        "metrics_smoothing_episodes": 5000,

        # === Model ===
        # Use two Q-networks (instead of one) for action-value estimation.
        # Note: Each Q-network will have its own target network.
        "twin_q": True,
        # Use a e.g. conv2D state preprocessing network before concatenating the
        # resulting (feature) vector with the action input for the input to
        # the Q-networks.
        "use_state_preprocessor": False,
        # Model options for the Q network(s).
        "Q_model": {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128, 128],
        },
        # Model options for the policy function.
        "policy_model": {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128, 128],
        },
        # Unsquash actions to the upper and lower bounds of env's action space.
        # Ignored for discrete action spaces.
        "normalize_actions": False,

        # === Learning ===
        # Disable setting done=True at end of episode. This should be set to True
        # for infinite-horizon MDPs (e.g., many continuous control problems).
        "no_done_at_end": False,
        # Update the target by \tau * policy + (1-\tau) * target_policy.
        "tau": choice([5e-3, 5e-2, 5e-4]),
        # Initial value to use for the entropy weight alpha.
        "initial_alpha": choice([1.0, 0.1, 0.01, 0.001]),
        # Target entropy lower bound. If "auto", will be set to -|A| (e.g. -2.0 for
        # Discrete(2), -3.0 for Box(shape=(3,))).
        # This is the inverse of reward scale, and will be optimized automatically.
        "target_entropy": choice([None, -0.5, -0.25]),
        # N-step target updates. If >1, sars' tuples in trajectories will be
        # postprocessed to become sa[discounted sum of R][s t+n] tuples.
        "n_step": 1,
        # Number of env steps to optimize for before returning.
        "timesteps_per_iteration": 100,

        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": choice([int(1e6), int(1e5), int(2e5)]),
        # If True prioritized replay buffer will be used.
        "prioritized_replay": choice([False, True]),
        "prioritized_replay_alpha": 0.6,
        "prioritized_replay_beta": 0.4,
        "prioritized_replay_eps": 1e-6,
        "prioritized_replay_beta_annealing_timesteps": 20000,
        "final_prioritized_replay_beta": 0.4,
        # Whether to LZ4 compress observations
        "compress_observations": True,
        # If set, this will fix the ratio of replayed from a buffer and learned on
        # timesteps to sampled from an environment and stored in the replay buffer
        # timesteps. Otherwise, the replay will proceed at the native ratio
        # determined by (train_batch_size / rollout_fragment_length).
        "training_intensity": None,

        # === Optimization ===
        "optimization": choice([
            {"actor_learning_rate": 3e-4, "critic_learning_rate": 3e-4, "entropy_learning_rate": 3e-4},
            {"actor_learning_rate": 3e-3, "critic_learning_rate": 3e-3, "entropy_learning_rate": 3e-3},
            {"actor_learning_rate": 3e-2, "critic_learning_rate": 3e-2, "entropy_learning_rate": 3e-2},
            {"actor_learning_rate": 3e-5, "critic_learning_rate": 3e-5, "entropy_learning_rate": 3e-5},
        ]),
        # If not None, clip gradients during optimization at this value.
        "grad_clip": None,
        # How many steps of the model to sample before learning starts.
        "learning_starts": 1500,
        # Update the replay buffer with this many samples at once. Note that this
        # setting applies per-worker if num_workers > 1.
        "rollout_fragment_length": choice([1, 2, 4, 8, 16]),
        # Size of a batched sampled from replay buffer for training. Note that
        # if async_updates is set, then each worker returns gradients for a
        # batch of this size.
        "train_batch_size": choice([256, 512, 1024, 2048]),
        # Update the target network every `target_network_update_freq` steps.
        "target_network_update_freq": 0,

    }

    search = HyperOptSearch(metric="br_reward_mean", mode="max", n_initial_points=50)

    tune.run(run_or_experiment=trainer_class,
             name=experiment_name,
             metric="br_reward_mean",

             config=hyperparams,
             num_samples=2000000,
             search_alg=search,
             mode="max",
             local_dir=data_dir(),
             stop={"timesteps_total": int(5e5)},
             loggers=[get_trainer_logger_creator(
                 base_dir=data_dir(),
                 scenario_name=experiment_name,
                 should_log_result_fn=lambda result: result["training_iteration"] % 20 == 0)],
             )
