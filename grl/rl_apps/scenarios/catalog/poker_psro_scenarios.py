from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo import PPOTrainer, PPOTorchPolicy

from grl.envs.poker_multi_agent_env import PokerMultiAgentEnv
from grl.rl_apps.scenarios.psro_scenario import PSROScenario
from grl.rl_apps.scenarios.catalog import scenario_catalog
from grl.rl_apps.scenarios.catalog.common import default_if_creating_ray_head
from grl.rl_apps.scenarios.trainer_configs.poker_psro_configs import *
from grl.rllib_tools.modified_policies.simple_q_torch_policy import SimpleQTorchPolicyPatched
from grl.rl_apps.scenarios.stopping_conditions import *

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
    trainer_class=DQNTrainer,
    policy_classes={
        "metanash": SimpleQTorchPolicyPatched,
        "best_response": SimpleQTorchPolicyPatched,
        "eval": SimpleQTorchPolicyPatched,
    },
    num_eval_workers=8,
    games_per_payoff_eval=20000,
    p2sro=False,
    get_trainer_config=psro_kuhn_dqn_params,
    psro_get_stopping_condition=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_episodes=int(2e4),
        check_plateau_every_n_episodes=int(2e4),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_episodes=int(1e5),
    ),
))

scenario_catalog.add(PSROScenario(
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
    trainer_class=DQNTrainer,
    policy_classes={
        "metanash": SimpleQTorchPolicyPatched,
        "best_response": SimpleQTorchPolicyPatched,
        "eval": SimpleQTorchPolicyPatched,
    },
    num_eval_workers=8,
    games_per_payoff_eval=3000,
    p2sro=False,
    get_trainer_config=psro_leduc_dqn_params,
    psro_get_stopping_condition=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_episodes=int(2e4),
        check_plateau_every_n_episodes=int(2e4),
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
        trainer_class=DQNTrainer,
        policy_classes={
            "metanash": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
            "eval": SimpleQTorchPolicyPatched,
        },
        num_eval_workers=8,
        games_per_payoff_eval=3000,
        p2sro=False,
        get_trainer_config=psro_leduc_dqn_params,
        psro_get_stopping_condition=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),
    ))

# 12_no_limit_leduc_psro_dqn
# 30_no_limit_leduc_psro_dqn
# 60_no_limit_leduc_psro_dqn

# 1000_no_limit_leduc_psro_ppo

for stack_size in [12, 30, 60, 100, 1000]:
    scenario_catalog.add(PSROScenario(
        name=f"{stack_size}_no_limit_leduc_psro_dqn",
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
        trainer_class=DQNTrainer,
        policy_classes={
            "metanash": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
            "eval": SimpleQTorchPolicyPatched,
        },
        num_eval_workers=8,
        games_per_payoff_eval=3000,
        p2sro=False,
        get_trainer_config=psro_leduc_dqn_params,
        psro_get_stopping_condition=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),
    ))

    scenario_catalog.add(PSROScenario(
        name=f"{stack_size}_no_limit_leduc_psro_ppo",
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
        trainer_class=PPOTrainer,
        policy_classes={
            "metanash": PPOTorchPolicy,
            "best_response": PPOTorchPolicy,
            "eval": PPOTorchPolicy,
        },
        num_eval_workers=8,
        games_per_payoff_eval=10000,
        p2sro=False,
        get_trainer_config=psro_leduc_ppo_params,
        psro_get_stopping_condition=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),
    ))