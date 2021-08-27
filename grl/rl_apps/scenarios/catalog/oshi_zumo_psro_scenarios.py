from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo import PPOTrainer, PPOTorchPolicy

from grl.envs.oshi_zumo_multi_agent_env import OshiZumoMultiAgentEnv, TinyOshiZumoMultiAgentEnv, \
    ThousandActionOshiZumoMultiAgentEnv, MediumOshiZumoMultiAgentEnv
from grl.rl_apps.scenarios.catalog import scenario_catalog
from grl.rl_apps.scenarios.catalog.common import default_if_creating_ray_head
from grl.rl_apps.scenarios.psro_scenario import PSROScenario
from grl.rl_apps.scenarios.stopping_conditions import *
from grl.rl_apps.scenarios.trainer_configs.oshi_zumo_configs import *
from grl.rl_apps.scenarios.trainer_configs.poker_psro_configs import *
from grl.rllib_tools.modified_policies.simple_q_torch_policy import SimpleQTorchPolicyPatched

orig = PSROScenario(
    name=f"oshi_zumo_psro_dqn",
    ray_cluster_cpus=default_if_creating_ray_head(default=8),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=1,
    env_class=OshiZumoMultiAgentEnv,
    env_config={

    },
    mix_metanash_with_uniform_dist_coeff=0.0,
    allow_stochastic_best_responses=False,
    trainer_class=DQNTrainer,
    policy_classes={
        "metanash": SimpleQTorchPolicyPatched,
        "best_response": SimpleQTorchPolicyPatched,
        "eval": SimpleQTorchPolicyPatched,
    },
    num_eval_workers=1,
    games_per_payoff_eval=1,
    p2sro=False,
    p2sro_payoff_table_exponential_avg_coeff=None,
    p2sro_sync_with_payoff_table_every_n_episodes=None,
    single_agent_symmetric_game=False,
    get_trainer_config=psro_leduc_dqn_params,
    psro_get_stopping_condition=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_episodes=int(2e4),
        check_plateau_every_n_episodes=int(2e4),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_episodes=int(1e5),
    ),
    calc_exploitability_for_openspiel_env=False,
)
scenario_catalog.add(orig)

faster_stop_cond = orig.with_updates(
    name=f"oshi_zumo_psro_dqn_faster_stop_cond",
    psro_get_stopping_condition=lambda: TimeStepsSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_steps=int(100e3),
        check_plateau_every_n_steps=int(50e3),
        minimum_reward_improvement_otherwise_plateaued=0.02,
        max_train_steps=int(1e6),
    ),
)
scenario_catalog.add(faster_stop_cond)

ppo_oshi = PSROScenario(
    name=f"1000_oshi_zumo_psro_ppo",
    ray_cluster_cpus=default_if_creating_ray_head(default=8),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=1,
    env_class=ThousandActionOshiZumoMultiAgentEnv,
    env_config={
        "append_valid_actions_mask_to_obs": False,
        "continuous_action_space": True,
    },
    mix_metanash_with_uniform_dist_coeff=0.0,
    allow_stochastic_best_responses=False,
    trainer_class=PPOTrainer,
    policy_classes={
        "metanash": PPOTorchPolicy,
        "best_response": PPOTorchPolicy,
        "eval": PPOTorchPolicy,
    },
    num_eval_workers=8,
    games_per_payoff_eval=1,
    p2sro=False,
    p2sro_payoff_table_exponential_avg_coeff=None,
    p2sro_sync_with_payoff_table_every_n_episodes=None,
    single_agent_symmetric_game=False,
    get_trainer_config=psro_oshi_ppo_params,
    psro_get_stopping_condition=lambda: TimeStepsSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_steps=int(100e3),
        check_plateau_every_n_steps=int(50e3),
        minimum_reward_improvement_otherwise_plateaued=0.02,
        max_train_steps=int(1e6),
    ),
    calc_exploitability_for_openspiel_env=False,
)
scenario_catalog.add(ppo_oshi)

scenario_catalog.add(ppo_oshi.with_updates(
    name=f"1000_oshi_zumo_psro_ppo_larger",
    get_trainer_config=larger_psro_oshi_ppo_params,
    num_eval_workers=2,
    games_per_payoff_eval=1,
    psro_get_stopping_condition=lambda: TimeStepsSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_steps=int(300e3),
        check_plateau_every_n_steps=int(100e3),
        must_be_non_negative_reward=True,
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_steps=int(3e6),
    ),
))

scenario_catalog.add(ppo_oshi.with_updates(
    name=f"1000_oshi_zumo_psro_ppo_larger_lower_entropy",
    get_trainer_config=larger_psro_oshi_ppo_params_lower_entropy,
    num_eval_workers=2,
    games_per_payoff_eval=1,
    psro_get_stopping_condition=lambda: TimeStepsSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_steps=int(200e3),
        check_plateau_every_n_steps=int(100e3),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_steps=int(2e6),
    ),
))

scenario_catalog.add(PSROScenario(
    name=f"oshi_zumo_tiny_psro_dqn",
    ray_cluster_cpus=default_if_creating_ray_head(default=8),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=1,
    env_class=TinyOshiZumoMultiAgentEnv,
    env_config={

    },
    mix_metanash_with_uniform_dist_coeff=0.0,
    allow_stochastic_best_responses=False,
    trainer_class=DQNTrainer,
    policy_classes={
        "metanash": SimpleQTorchPolicyPatched,
        "best_response": SimpleQTorchPolicyPatched,
        "eval": SimpleQTorchPolicyPatched,
    },
    num_eval_workers=1,
    games_per_payoff_eval=1,
    p2sro=False,
    p2sro_payoff_table_exponential_avg_coeff=None,
    p2sro_sync_with_payoff_table_every_n_episodes=None,
    single_agent_symmetric_game=False,
    get_trainer_config=tiny_oshi_zumo_psro_dqn_params,
    psro_get_stopping_condition=lambda: TimeStepsSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_steps=int(40e3),
        check_plateau_every_n_steps=int(20e3),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_steps=int(200e3),
    ),
    calc_exploitability_for_openspiel_env=True,
))

scenario_catalog.add(PSROScenario(
    name=f"oshi_zumo_medium_psro_dqn",
    ray_cluster_cpus=default_if_creating_ray_head(default=8),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=1,
    env_class=MediumOshiZumoMultiAgentEnv,
    env_config={

    },
    mix_metanash_with_uniform_dist_coeff=0.0,
    allow_stochastic_best_responses=False,
    trainer_class=DQNTrainer,
    policy_classes={
        "metanash": SimpleQTorchPolicyPatched,
        "best_response": SimpleQTorchPolicyPatched,
        "eval": SimpleQTorchPolicyPatched,
    },
    num_eval_workers=1,
    games_per_payoff_eval=1,
    p2sro=False,
    p2sro_payoff_table_exponential_avg_coeff=None,
    p2sro_sync_with_payoff_table_every_n_episodes=None,
    single_agent_symmetric_game=False,
    get_trainer_config=medium_oshi_zumo_psro_dqn_params,
    psro_get_stopping_condition=lambda: TimeStepsSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_steps=int(800e3),
        check_plateau_every_n_steps=int(400e3),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_steps=int(5e6),
    ),
    calc_exploitability_for_openspiel_env=False,
))
