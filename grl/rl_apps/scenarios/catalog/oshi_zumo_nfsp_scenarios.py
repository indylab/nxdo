from ray.rllib.agents.dqn import DQNTrainer

from grl.algos.nfsp_rllib.nfsp import NFSPTrainer, NFSPTorchAveragePolicy
from grl.envs.oshi_zumo_multi_agent_env import OshiZumoMultiAgentEnv, ThousandActionOshiZumoMultiAgentEnv, \
    TinyOshiZumoMultiAgentEnv, MediumOshiZumoMultiAgentEnv
from grl.rl_apps.scenarios.catalog import scenario_catalog
from grl.rl_apps.scenarios.catalog.common import default_if_creating_ray_head
from grl.rl_apps.scenarios.nfsp_scenario import NFSPScenario
from grl.rl_apps.scenarios.stopping_conditions import NoStoppingCondition
from grl.rl_apps.scenarios.trainer_configs.oshi_zumo_configs import *
from grl.rl_apps.scenarios.trainer_configs.poker_nfsp_configs import *
from grl.rllib_tools.modified_policies.simple_q_torch_policy import SimpleQTorchPolicyPatched

scenario_catalog.add(NFSPScenario(
    name=f"oshi_zumo_nfsp_dqn",
    ray_cluster_cpus=default_if_creating_ray_head(default=4),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=1,
    env_class=OshiZumoMultiAgentEnv,
    env_config={

    },
    trainer_class=DQNTrainer,
    avg_trainer_class=NFSPTrainer,
    policy_classes={
        "average_policy": NFSPTorchAveragePolicy,
        "best_response": SimpleQTorchPolicyPatched,
    },
    get_trainer_config=nfsp_leduc_dqn_params,
    get_avg_trainer_config=nfsp_leduc_avg_policy_params,
    anticipatory_param=0.1,
    nfsp_get_stopping_condition=lambda: NoStoppingCondition(),
    calculate_openspiel_metanash=False,
    calculate_openspiel_metanash_at_end=False,
    calc_metanash_every_n_iters=100,
    checkpoint_every_n_iters=1000
))

scenario_catalog.add(NFSPScenario(
    name=f"oshi_zumo_tiny_nfsp_dqn",
    ray_cluster_cpus=default_if_creating_ray_head(default=4),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=1,
    env_class=TinyOshiZumoMultiAgentEnv,
    env_config={

    },
    trainer_class=DQNTrainer,
    avg_trainer_class=NFSPTrainer,
    policy_classes={
        "average_policy": NFSPTorchAveragePolicy,
        "best_response": SimpleQTorchPolicyPatched,
    },
    get_trainer_config=tiny_oshi_zumo_nfsp_dqn_params,
    get_avg_trainer_config=tiny_oshi_zumo_nfsp_avg_policy_params,
    anticipatory_param=0.1,
    nfsp_get_stopping_condition=lambda: NoStoppingCondition(),
    calculate_openspiel_metanash=True,
    calculate_openspiel_metanash_at_end=False,
    calc_metanash_every_n_iters=1000,
    checkpoint_every_n_iters=1000
))

scenario_catalog.add(NFSPScenario(
    name=f"oshi_zumo_medium_nfsp_dqn",
    ray_cluster_cpus=default_if_creating_ray_head(default=4),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=1,
    env_class=MediumOshiZumoMultiAgentEnv,
    env_config={

    },
    trainer_class=DQNTrainer,
    avg_trainer_class=NFSPTrainer,
    policy_classes={
        "average_policy": NFSPTorchAveragePolicy,
        "best_response": SimpleQTorchPolicyPatched,
    },
    get_trainer_config=medium_oshi_zumo_nfsp_dqn_params,
    get_avg_trainer_config=medium_oshi_zumo_nfsp_avg_policy_params,
    anticipatory_param=0.1,
    nfsp_get_stopping_condition=lambda: NoStoppingCondition(),
    calculate_openspiel_metanash=False,
    calculate_openspiel_metanash_at_end=False,
    calc_metanash_every_n_iters=1000,
    checkpoint_every_n_iters=1000
))

thousand_oshi = NFSPScenario(
    name=f"1000_oshi_zumo_nfsp_dqn",
    ray_cluster_cpus=default_if_creating_ray_head(default=4),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=1,
    env_class=ThousandActionOshiZumoMultiAgentEnv,
    env_config={

    },
    trainer_class=DQNTrainer,
    avg_trainer_class=NFSPTrainer,
    policy_classes={
        "average_policy": NFSPTorchAveragePolicy,
        "best_response": SimpleQTorchPolicyPatched,
    },
    get_trainer_config=nfsp_leduc_dqn_params,
    get_avg_trainer_config=nfsp_leduc_avg_policy_params,
    anticipatory_param=0.1,
    nfsp_get_stopping_condition=lambda: NoStoppingCondition(),
    calculate_openspiel_metanash=False,
    calculate_openspiel_metanash_at_end=False,
    calc_metanash_every_n_iters=100,
    checkpoint_every_n_iters=1000
)
scenario_catalog.add(thousand_oshi)

scenario_catalog.add(thousand_oshi.with_updates(
    name=f"1000_oshi_zumo_nfsp_larger_dqn_larger",
    get_trainer_config=larger_nfsp_oshi_ppo_params_two_layers_with_valid_actions_model,
    get_avg_trainer_config=larger_nfsp_oshi_ppo_avg_policy_params_two_layers_with_valid_actions_model,
))
