from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo import PPOTrainer, PPOTorchPolicy

from grl.algos.nfsp_rllib.nfsp import NFSPTrainer, NFSPTorchAveragePolicy
from grl.envs.attack_and_counter_game import AttackCounterGameMultiAgentEnv
from grl.rl_apps.nxdo.solve_restricted_game_fns import *
from grl.rl_apps.scenarios.catalog import scenario_catalog
from grl.rl_apps.scenarios.catalog.common import default_if_creating_ray_head
from grl.rl_apps.scenarios.nfsp_scenario import NFSPScenario
from grl.rl_apps.scenarios.psro_scenario import PSROScenario
from grl.rl_apps.scenarios.stopping_conditions import *
from grl.rl_apps.scenarios.stopping_conditions import NoStoppingCondition
from grl.rl_apps.scenarios.trainer_configs.loss_game_configs import *
from grl.rllib_tools.modified_policies.simple_q_torch_policy import SimpleQTorchPolicyPatched

for total_moves in [1]:
    for game_class in [AttackCounterGameMultiAgentEnv]:
        scenario_catalog.add(PSROScenario(
            name=f"attack_and_counter_game_psro",
            ray_cluster_cpus=default_if_creating_ray_head(default=11),
            ray_cluster_gpus=default_if_creating_ray_head(default=0),
            ray_object_store_memory_cap_gigabytes=1,
            env_class=game_class,
            env_config={},
            mix_metanash_with_uniform_dist_coeff=0.0,
            allow_stochastic_best_responses=True,
            trainer_class=PPOTrainer,
            policy_classes={
                "metanash": PPOTorchPolicy,
                "best_response": PPOTorchPolicy,
                "eval": PPOTorchPolicy,
            },
            num_eval_workers=10,
            games_per_payoff_eval=1000,
            p2sro=False,
            p2sro_payoff_table_exponential_avg_coeff=None,
            p2sro_sync_with_payoff_table_every_n_episodes=None,
            single_agent_symmetric_game=False,
            get_trainer_config=attack_and_counter_psro_ppo_params,
            psro_get_stopping_condition=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
                br_policy_id="best_response",
                dont_check_plateau_before_n_episodes=int(2e4),
                check_plateau_every_n_episodes=int(2e4),
                minimum_reward_improvement_otherwise_plateaued=0.01,
                max_train_episodes=int(1e5),
            ),
            calc_exploitability_for_openspiel_env=False,
        ))

        scenario_catalog.add(NXDOScenario(
            name=f"attack_and_counter_game_nxdo",
            ray_cluster_cpus=default_if_creating_ray_head(default=8),
            ray_cluster_gpus=default_if_creating_ray_head(default=0),
            ray_object_store_memory_cap_gigabytes=1,
            use_openspiel_restricted_game=False,
            get_restricted_game_custom_model=lambda env: None,
            xdo_metanash_method="nfsp",
            allow_stochastic_best_responses=True,
            get_restricted_game_solver=lambda scenario: SolveRestrictedGameIncreasingTimeToSolve(
                scenario=scenario,
                dont_solve_first_n_nxdo_iters=5,
                increase_multiplier=1.5,
                starting_steps=int(1e6),
            ),
            env_class=game_class,
            env_config={},

            trainer_class_br=PPOTrainer,
            policy_classes_br={
                "metanash": NFSPTorchAveragePolicy,
                "best_response": PPOTorchPolicy,
            },
            get_trainer_config_br=attack_and_counter_psro_ppo_params,
            get_stopping_condition_br=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
                br_policy_id="best_response",
                dont_check_plateau_before_n_episodes=int(2e4),
                check_plateau_every_n_episodes=int(2e4),
                minimum_reward_improvement_otherwise_plateaued=0.01,
                max_train_episodes=int(1e5),
            ),
            trainer_class_nfsp=DQNTrainer,
            avg_trainer_class_nfsp=NFSPTrainer,
            policy_classes_nfsp={
                "average_policy": NFSPTorchAveragePolicy,
                "delegate_policy": PPOTorchPolicy,
                "best_response": SimpleQTorchPolicyPatched,
            },
            anticipatory_param_nfsp=0.1,
            get_trainer_config_nfsp=loss_game_nfsp_dqn_params_orig,
            get_avg_trainer_config_nfsp=loss_game_nfsp_avg_policy_params_orig,
            calculate_openspiel_metanash=False,
            calculate_openspiel_metanash_at_end=False,
            calc_metanash_every_n_iters=0,
            metanash_metrics_smoothing_episodes_override=50000,
        ))

        scenario_catalog.add(NFSPScenario(
            name=f"attack_and_counter_game_nfsp",
            ray_cluster_cpus=default_if_creating_ray_head(default=4),
            ray_cluster_gpus=default_if_creating_ray_head(default=0),
            ray_object_store_memory_cap_gigabytes=2,
            env_class=game_class,
            trainer_class=DQNTrainer,
            avg_trainer_class=NFSPTrainer,
            policy_classes={
                "average_policy": NFSPTorchAveragePolicy,
                "best_response": SimpleQTorchPolicyPatched,
            },
            get_trainer_config=loss_game_nfsp_dqn_params,
            get_avg_trainer_config=loss_game_nfsp_avg_policy_params,
            anticipatory_param=0.1,
            nfsp_get_stopping_condition=lambda: NoStoppingCondition(),
            calculate_openspiel_metanash=False,
            calculate_openspiel_metanash_at_end=False,
            calc_metanash_every_n_iters=0,
            checkpoint_every_n_iters=500,
            env_config={
                "discrete_actions_for_players": [0, 1],
            },
        ))
