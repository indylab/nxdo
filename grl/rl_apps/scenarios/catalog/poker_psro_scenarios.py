from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo import PPOTrainer, PPOTorchPolicy

from grl.envs.poker_multi_agent_env import PokerMultiAgentEnv
from grl.rl_apps.scenarios.catalog import scenario_catalog
from grl.rl_apps.scenarios.catalog.common import default_if_creating_ray_head
from grl.rl_apps.scenarios.psro_scenario import PSROScenario
from grl.rl_apps.scenarios.stopping_conditions import *
from grl.rl_apps.scenarios.trainer_configs.poker_psro_configs import *
from grl.rllib_tools.modified_policies.simple_q_torch_policy import SimpleQTorchPolicyPatched

scenario_catalog.add(PSROScenario(
    name="kuhn_psro_dqn",
    ray_cluster_cpus=default_if_creating_ray_head(default=8),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=1,
    env_class=PokerMultiAgentEnv,
    env_config={
        "version": "kuhn_poker",
        "fixed_players": True,
    },
    mix_metanash_with_uniform_dist_coeff=0.0,
    allow_stochastic_best_responses=False,
    trainer_class=DQNTrainer,
    policy_classes={
        "metanash": SimpleQTorchPolicyPatched,
        "best_response": SimpleQTorchPolicyPatched,
        "eval": SimpleQTorchPolicyPatched,
    },
    num_eval_workers=8,
    games_per_payoff_eval=20000,
    p2sro=False,
    p2sro_payoff_table_exponential_avg_coeff=None,
    p2sro_sync_with_payoff_table_every_n_episodes=None,
    single_agent_symmetric_game=False,
    get_trainer_config=psro_kuhn_dqn_params,
    psro_get_stopping_condition=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_episodes=int(2e4),
        check_plateau_every_n_episodes=int(2e4),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_episodes=int(1e5),
    ),
    calc_exploitability_for_openspiel_env=True,
))

leduc_psro_dqn = PSROScenario(
    name="leduc_psro_dqn",
    ray_cluster_cpus=default_if_creating_ray_head(default=8),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=1,
    env_class=PokerMultiAgentEnv,
    env_config={
        'version': "leduc_poker",
        "fixed_players": True,
        "append_valid_actions_mask_to_obs": True,
    },
    mix_metanash_with_uniform_dist_coeff=0.0,
    allow_stochastic_best_responses=False,
    trainer_class=DQNTrainer,
    policy_classes={
        "metanash": SimpleQTorchPolicyPatched,
        "best_response": SimpleQTorchPolicyPatched,
        "eval": SimpleQTorchPolicyPatched,
    },
    num_eval_workers=8,
    games_per_payoff_eval=3000,
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
    calc_exploitability_for_openspiel_env=True,
)
scenario_catalog.add(leduc_psro_dqn)

scenario_catalog.add(leduc_psro_dqn.with_updates(
    name="symmetric_leduc_psro_dqn",
    env_config={
        'version': "leduc_poker",
        "fixed_players": False,
        "append_valid_actions_mask_to_obs": True,
    },
    mix_metanash_with_uniform_dist_coeff=0.1,
    single_agent_symmetric_game=True,
    psro_get_stopping_condition=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_episodes=int(5e3),
        check_plateau_every_n_episodes=int(5e3),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_episodes=int(1e5),
    ),
))

scenario_catalog.add(leduc_psro_dqn.with_updates(
    name="symmetric_leduc_p2sro_dqn_3_br_learners",
    ray_cluster_cpus=default_if_creating_ray_head(default=4 * 3),  # 4 cpus for each of 3 workers
    env_config={
        'version': "leduc_poker",
        "fixed_players": False,
        "append_valid_actions_mask_to_obs": True,
    },
    mix_metanash_with_uniform_dist_coeff=0.1,
    p2sro=True,
    p2sro_payoff_table_exponential_avg_coeff=1.0 / 3000,
    p2sro_sync_with_payoff_table_every_n_episodes=100,
    single_agent_symmetric_game=True,
    psro_get_stopping_condition=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_episodes=int(5e3),
        check_plateau_every_n_episodes=int(5e3),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_episodes=int(1e5),
    ),
))

scenario_catalog.add(leduc_psro_dqn.with_updates(
    name="symmetric_leduc_p2sro_dqn_3_br_learners_fast_update",
    ray_cluster_cpus=default_if_creating_ray_head(default=4 * 3),  # 4 cpus for each of 3 workers
    env_config={
        'version': "leduc_poker",
        "fixed_players": False,
        "append_valid_actions_mask_to_obs": True,
    },
    mix_metanash_with_uniform_dist_coeff=0.1,
    p2sro=True,
    p2sro_payoff_table_exponential_avg_coeff=1.0 / 1000,
    p2sro_sync_with_payoff_table_every_n_episodes=100,
    single_agent_symmetric_game=True,
    psro_get_stopping_condition=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_episodes=int(5e3),
        check_plateau_every_n_episodes=int(5e3),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_episodes=int(1e5),
    ),
))

# 20_clone_leduc_psro_dqn
# 40_clone_leduc_psro_dqn
# 80_clone_leduc_psro_dqn
for dummy_action_multiplier in [20, 40, 80]:
    scenario_catalog.add(PSROScenario(
        name=f"{dummy_action_multiplier}_clone_leduc_psro_dqn",
        ray_cluster_cpus=default_if_creating_ray_head(default=8),
        ray_cluster_gpus=default_if_creating_ray_head(default=0),
        ray_object_store_memory_cap_gigabytes=1,
        env_class=PokerMultiAgentEnv,
        env_config={
            'version': "leduc_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "dummy_action_multiplier": dummy_action_multiplier,
        },
        mix_metanash_with_uniform_dist_coeff=0.0,
        allow_stochastic_best_responses=False,
        trainer_class=DQNTrainer,
        policy_classes={
            "metanash": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
            "eval": SimpleQTorchPolicyPatched,
        },
        num_eval_workers=8,
        games_per_payoff_eval=3000,
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
        calc_exploitability_for_openspiel_env=True,
    ))

# 12_no_limit_leduc_psro_dqn
# 30_no_limit_leduc_psro_dqn
# 60_no_limit_leduc_psro_dqn

# 1000_no_limit_leduc_psro_ppo
# 1000_no_limit_leduc_psro_ppo_stochastic

for stack_size in [12, 30, 60, 100, 1000]:
    for allow_stochastic_best_responses in [True, False]:
        stochastic_str = "_stochastic" if allow_stochastic_best_responses else ""

        scenario_catalog.add(PSROScenario(
            name=f"{stack_size}_no_limit_leduc_psro_dqn{stochastic_str}",
            ray_cluster_cpus=default_if_creating_ray_head(default=8),
            ray_cluster_gpus=default_if_creating_ray_head(default=0),
            ray_object_store_memory_cap_gigabytes=1,
            env_class=PokerMultiAgentEnv,
            env_config={
                'version': "universal_poker",
                "fixed_players": True,
                "append_valid_actions_mask_to_obs": True,
                "universal_poker_stack_size": stack_size,
            },
            mix_metanash_with_uniform_dist_coeff=0.0,
            allow_stochastic_best_responses=allow_stochastic_best_responses,
            trainer_class=DQNTrainer,
            policy_classes={
                "metanash": SimpleQTorchPolicyPatched,
                "best_response": SimpleQTorchPolicyPatched,
                "eval": SimpleQTorchPolicyPatched,
            },
            num_eval_workers=8,
            games_per_payoff_eval=3000,
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
        ))

        ppo_no_limit = PSROScenario(
            name=f"{stack_size}_no_limit_leduc_psro_ppo{stochastic_str}",
            ray_cluster_cpus=default_if_creating_ray_head(default=8),
            ray_cluster_gpus=default_if_creating_ray_head(default=0),
            ray_object_store_memory_cap_gigabytes=1,
            env_class=PokerMultiAgentEnv,
            env_config={
                'version': "universal_poker",
                "fixed_players": True,
                "append_valid_actions_mask_to_obs": False,
                "universal_poker_stack_size": stack_size,
                "continuous_action_space": True,
            },
            mix_metanash_with_uniform_dist_coeff=0.0,
            allow_stochastic_best_responses=allow_stochastic_best_responses,
            trainer_class=PPOTrainer,
            policy_classes={
                "metanash": PPOTorchPolicy,
                "best_response": PPOTorchPolicy,
                "eval": PPOTorchPolicy,
            },
            num_eval_workers=8,
            games_per_payoff_eval=10000,
            p2sro=False,
            p2sro_payoff_table_exponential_avg_coeff=None,
            p2sro_sync_with_payoff_table_every_n_episodes=None,
            single_agent_symmetric_game=False,
            get_trainer_config=psro_leduc_ppo_params,
            psro_get_stopping_condition=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
                br_policy_id="best_response",
                dont_check_plateau_before_n_episodes=int(2e4),
                check_plateau_every_n_episodes=int(2e4),
                minimum_reward_improvement_otherwise_plateaued=0.01,
                max_train_episodes=int(1e5),
            ),
            calc_exploitability_for_openspiel_env=False,
        )
        scenario_catalog.add(ppo_no_limit)

        scenario_catalog.add(ppo_no_limit.with_updates(
            name=f"{stack_size}_3_round_no_limit_leduc_psro_ppo{stochastic_str}",
            env_config={
                'version': "universal_poker",
                "fixed_players": True,
                "append_valid_actions_mask_to_obs": False,
                "universal_poker_stack_size": stack_size,
                "continuous_action_space": True,
                "universal_poker_num_rounds": 3,
            },
        ))
        scenario_catalog.add(ppo_no_limit.with_updates(
            name=f"{stack_size}_4_round_no_limit_leduc_psro_ppo{stochastic_str}",
            env_config={
                'version': "universal_poker",
                "fixed_players": True,
                "append_valid_actions_mask_to_obs": False,
                "universal_poker_stack_size": stack_size,
                "continuous_action_space": True,
                "universal_poker_num_rounds": 4,
            },
        ))
        scenario_catalog.add(ppo_no_limit.with_updates(
            name=f"{stack_size}_6_rank_no_limit_leduc_psro_ppo{stochastic_str}",
            env_config={
                'version': "universal_poker",
                "fixed_players": True,
                "append_valid_actions_mask_to_obs": False,
                "universal_poker_stack_size": stack_size,
                "continuous_action_space": True,
                "universal_poker_num_ranks": 6,
            },
        ))
