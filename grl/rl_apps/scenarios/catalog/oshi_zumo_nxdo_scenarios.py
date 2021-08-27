from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo import PPOTrainer, PPOTorchPolicy

from grl.algos.nfsp_rllib.nfsp import NFSPTrainer, NFSPTorchAveragePolicy
from grl.envs.oshi_zumo_multi_agent_env import OshiZumoMultiAgentEnv, TinyOshiZumoMultiAgentEnv, \
    ThousandActionOshiZumoMultiAgentEnv, MediumOshiZumoMultiAgentEnv
from grl.rl_apps.nxdo.solve_restricted_game_fns import *
from grl.rl_apps.scenarios.catalog import scenario_catalog
from grl.rl_apps.scenarios.catalog.common import default_if_creating_ray_head
from grl.rl_apps.scenarios.stopping_conditions import *
from grl.rl_apps.scenarios.trainer_configs.oshi_zumo_configs import *
from grl.rl_apps.scenarios.trainer_configs.poker_nfsp_configs import *
from grl.rl_apps.scenarios.trainer_configs.poker_psro_configs import *
from grl.rllib_tools.modified_policies.simple_q_torch_policy import SimpleQTorchPolicyPatched

for warm_start_iters in [0, 4, 5, 7, 15, 30, 40, 50]:
    dynamic_warm_start = NXDOScenario(
        name=f"oshi_zumo_nxdo_dqn_nfsp_dynamic_threshold_1_warm_start_{warm_start_iters}",
        ray_cluster_cpus=default_if_creating_ray_head(default=8),
        ray_cluster_gpus=default_if_creating_ray_head(default=0),
        ray_object_store_memory_cap_gigabytes=1,
        use_openspiel_restricted_game=False,
        get_restricted_game_custom_model=lambda env: None,
        xdo_metanash_method="nfsp",
        allow_stochastic_best_responses=False,
        get_restricted_game_solver=lambda scenario, warm_start_iters=warm_start_iters:
        SolveRestrictedGameDynamicRewardThreshold1(
            scenario=scenario,
            dont_solve_first_n_nxdo_iters=warm_start_iters,
            starting_rew_threshold=1.0,
            min_rew_threshold=0.05,
            min_episodes=50000,
            epsilon=0.05,
            required_fields=[],
        ),
        env_class=OshiZumoMultiAgentEnv,
        env_config={

        },
        trainer_class_br=DQNTrainer,
        policy_classes_br={
            "metanash": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },
        get_trainer_config_br=psro_leduc_dqn_params,
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
            "delegate_policy": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
        },
        anticipatory_param_nfsp=0.1,
        get_trainer_config_nfsp=nfsp_leduc_dqn_params,
        get_avg_trainer_config_nfsp=nfsp_leduc_avg_policy_params,
        calculate_openspiel_metanash=False,
        calculate_openspiel_metanash_at_end=False,
        calc_metanash_every_n_iters=50,
        metanash_metrics_smoothing_episodes_override=50000
    )
    scenario_catalog.add(dynamic_warm_start)

# oshi_zumo_tiny_nxdo_dqn_nfsp_warm_start_7
for warm_start_iters in [0, 4, 5, 7, 8, 9, 12, 15, 30, 40, 50]:
    # oshi_zumo_tiny_nxdo_dqn_nfsp_warm_start_7_simple_annealing
    scenario_catalog.add(NXDOScenario(
        name=f"oshi_zumo_tiny_nxdo_dqn_nfsp_warm_start_{warm_start_iters}_simple_annealing",
        ray_cluster_cpus=default_if_creating_ray_head(default=8),
        ray_cluster_gpus=default_if_creating_ray_head(default=0),
        ray_object_store_memory_cap_gigabytes=1,
        use_openspiel_restricted_game=False,
        get_restricted_game_custom_model=lambda env: None,
        xdo_metanash_method="nfsp",
        allow_stochastic_best_responses=False,
        get_restricted_game_solver=lambda scenario, warm_start_iters=warm_start_iters:
        SolveRestrictedGameDynamicRewardThreshold1(
            scenario=scenario,
            dont_solve_first_n_nxdo_iters=warm_start_iters,
            starting_rew_threshold=0.3,
            min_rew_threshold=0.0,
            min_episodes=None,
            min_steps=400000,
            decrease_threshold_every_iter=True,
            epsilon=0.0,
            required_fields=[],
        ),
        env_class=TinyOshiZumoMultiAgentEnv,
        env_config={

        },
        trainer_class_br=DQNTrainer,
        policy_classes_br={
            "metanash": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },
        get_trainer_config_br=tiny_oshi_zumo_psro_dqn_params,
        get_stopping_condition_br=lambda: TimeStepsSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_steps=int(40e3),
            check_plateau_every_n_steps=int(20e3),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_steps=int(200e3),
        ),
        trainer_class_nfsp=DQNTrainer,
        avg_trainer_class_nfsp=NFSPTrainer,
        policy_classes_nfsp={
            "average_policy": NFSPTorchAveragePolicy,
            "delegate_policy": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
        },
        anticipatory_param_nfsp=0.1,
        get_trainer_config_nfsp=tiny_oshi_zumo_nfsp_dqn_params,
        get_avg_trainer_config_nfsp=tiny_oshi_zumo_nfsp_avg_policy_params,
        calculate_openspiel_metanash=False,
        calculate_openspiel_metanash_at_end=True,
        calc_metanash_every_n_iters=50,
        metanash_metrics_smoothing_episodes_override=50000
    ))
    # oshi_zumo_medium_nxdo_dqn_nfsp_warm_start_7
    med_oshi = NXDOScenario(
        name=f"oshi_zumo_medium_nxdo_dqn_nfsp_warm_start_{warm_start_iters}",
        ray_cluster_cpus=default_if_creating_ray_head(default=8),
        ray_cluster_gpus=default_if_creating_ray_head(default=0),
        ray_object_store_memory_cap_gigabytes=1,
        use_openspiel_restricted_game=False,
        get_restricted_game_custom_model=lambda env: None,
        xdo_metanash_method="nfsp",
        allow_stochastic_best_responses=False,
        get_restricted_game_solver=lambda scenario, warm_start_iters=warm_start_iters:
        SolveRestrictedGameDynamicRewardThreshold1(
            scenario=scenario,
            dont_solve_first_n_nxdo_iters=warm_start_iters,
            starting_rew_threshold=1.0,
            min_rew_threshold=0.0,
            min_episodes=None,
            min_steps=400000,
            epsilon=0.05,
            required_fields=[],
        ),
        env_class=MediumOshiZumoMultiAgentEnv,
        env_config={

        },
        trainer_class_br=DQNTrainer,
        policy_classes_br={
            "metanash": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },
        get_trainer_config_br=medium_oshi_zumo_psro_dqn_params,
        get_stopping_condition_br=lambda: TimeStepsSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_steps=int(800e3),
            check_plateau_every_n_steps=int(400e3),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_steps=int(5e6),
        ),
        trainer_class_nfsp=DQNTrainer,
        avg_trainer_class_nfsp=NFSPTrainer,
        policy_classes_nfsp={
            "average_policy": NFSPTorchAveragePolicy,
            "delegate_policy": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
        },
        anticipatory_param_nfsp=0.1,
        get_trainer_config_nfsp=medium_oshi_zumo_nfsp_dqn_params,
        get_avg_trainer_config_nfsp=medium_oshi_zumo_nfsp_avg_policy_params,
        calculate_openspiel_metanash=False,
        calculate_openspiel_metanash_at_end=False,
        calc_metanash_every_n_iters=50,
        metanash_metrics_smoothing_episodes_override=50000
    )
    scenario_catalog.add(med_oshi)
    # oshi_zumo_medium_nxdo_dqn_nfsp_warm_start_7_simple_annealing
    # oshi_zumo_medium_nxdo_dqn_nfsp_warm_start_7_simple_annealing
    scenario_catalog.add(med_oshi.with_updates(
        name=f"oshi_zumo_medium_nxdo_dqn_nfsp_warm_start_{warm_start_iters}_simple_annealing",
        get_restricted_game_solver=lambda scenario, warm_start_iters=warm_start_iters:
        SolveRestrictedGameDynamicRewardThreshold1(
            scenario=scenario,
            dont_solve_first_n_nxdo_iters=warm_start_iters,
            starting_rew_threshold=1.0,
            min_rew_threshold=0.0,
            min_episodes=None,
            min_steps=500000,
            decrease_threshold_every_iter=True,
            epsilon=0.0,
            required_fields=[],
        ),
    ))

for warm_start_iters in [0, 4, 5, 7, 15, 30, 40, 50]:
    steps_400k = NXDOScenario(
        name=f"1000_oshi_zumo_nxdo_ppo_nfsp_dynamic_threshold_1_warm_start_{warm_start_iters}_nfsp_min_400k_steps",
        ray_cluster_cpus=default_if_creating_ray_head(default=8),
        ray_cluster_gpus=default_if_creating_ray_head(default=0),
        ray_object_store_memory_cap_gigabytes=1,
        use_openspiel_restricted_game=False,
        get_restricted_game_custom_model=None,
        xdo_metanash_method="nfsp",
        allow_stochastic_best_responses=False,
        get_restricted_game_solver=lambda scenario, warm_start_iters=warm_start_iters:
        SolveRestrictedGameDynamicRewardThreshold1(
            scenario=scenario,
            dont_solve_first_n_nxdo_iters=warm_start_iters,
            starting_rew_threshold=1.0,
            min_rew_threshold=0.0,
            min_episodes=None,
            min_steps=400000,
            epsilon=0.05,
            required_fields=[],
        ),
        env_class=ThousandActionOshiZumoMultiAgentEnv,
        env_config={
            "append_valid_actions_mask_to_obs": False,
            "continuous_action_space": True,
        },
        trainer_class_br=PPOTrainer,
        policy_classes_br={
            "metanash": NFSPTorchAveragePolicy,
            "best_response": PPOTorchPolicy,
        },
        get_trainer_config_br=larger_psro_oshi_ppo_params,
        get_stopping_condition_br=lambda: TimeStepsSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_steps=int(300e3),
            check_plateau_every_n_steps=int(100e3),
            must_be_non_negative_reward=True,
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_steps=int(3e6),
        ),
        trainer_class_nfsp=DQNTrainer,
        avg_trainer_class_nfsp=NFSPTrainer,
        policy_classes_nfsp={
            "average_policy": NFSPTorchAveragePolicy,
            "delegate_policy": PPOTorchPolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },
        anticipatory_param_nfsp=0.1,
        get_trainer_config_nfsp=larger_nfsp_oshi_ppo_params_two_layers_no_valid_actions_model,
        get_avg_trainer_config_nfsp=larger_nfsp_oshi_ppo_avg_policy_params_two_layers_no_valid_actions_model,
        calculate_openspiel_metanash=False,
        calculate_openspiel_metanash_at_end=False,
        calc_metanash_every_n_iters=50,
        metanash_metrics_smoothing_episodes_override=5000
    )
    scenario_catalog.add(steps_400k)

    # 1000_oshi_zumo_nxdo_ppo_nfsp_dynamic_threshold_1_warm_start_5_nfsp_min_400k_steps_lower_entropy
    scenario_catalog.add(NXDOScenario(
        name=f"1000_oshi_zumo_nxdo_ppo_nfsp_dynamic_threshold_1_warm_start_{warm_start_iters}_nfsp_min_400k_steps_lower_entropy",
        ray_cluster_cpus=default_if_creating_ray_head(default=8),
        ray_cluster_gpus=default_if_creating_ray_head(default=0),
        ray_object_store_memory_cap_gigabytes=1,
        use_openspiel_restricted_game=False,
        get_restricted_game_custom_model=None,
        xdo_metanash_method="nfsp",
        allow_stochastic_best_responses=False,
        get_restricted_game_solver=lambda scenario, warm_start_iters=warm_start_iters:
        SolveRestrictedGameDynamicRewardThreshold1(
            scenario=scenario,
            dont_solve_first_n_nxdo_iters=warm_start_iters,
            starting_rew_threshold=1.0,
            min_rew_threshold=0.0,
            min_episodes=None,
            min_steps=400000,
            epsilon=0.05,
            required_fields=[],
        ),
        env_class=ThousandActionOshiZumoMultiAgentEnv,
        env_config={
            "append_valid_actions_mask_to_obs": False,
            "continuous_action_space": True,
        },
        trainer_class_br=PPOTrainer,
        policy_classes_br={
            "metanash": NFSPTorchAveragePolicy,
            "best_response": PPOTorchPolicy,
        },
        get_trainer_config_br=larger_psro_oshi_ppo_params_lower_entropy,
        get_stopping_condition_br=lambda: TimeStepsSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_steps=int(200e3),
            check_plateau_every_n_steps=int(100e3),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_steps=int(2e6),
        ),
        trainer_class_nfsp=DQNTrainer,
        avg_trainer_class_nfsp=NFSPTrainer,
        policy_classes_nfsp={
            "average_policy": NFSPTorchAveragePolicy,
            "delegate_policy": PPOTorchPolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },
        anticipatory_param_nfsp=0.1,
        get_trainer_config_nfsp=larger_nfsp_oshi_ppo_params_two_layers_no_valid_actions_model,
        get_avg_trainer_config_nfsp=larger_nfsp_oshi_ppo_avg_policy_params_two_layers_no_valid_actions_model,
        calculate_openspiel_metanash=False,
        calculate_openspiel_metanash_at_end=False,
        calc_metanash_every_n_iters=50,
        metanash_metrics_smoothing_episodes_override=5000
    ))
