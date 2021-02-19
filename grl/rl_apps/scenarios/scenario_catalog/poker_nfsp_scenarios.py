from ray.rllib.agents.dqn import DQNTrainer

from grl.envs.poker_multi_agent_env import PokerMultiAgentEnv
from grl.nfsp_rllib.nfsp import NFSPTrainer, NFSPTorchAveragePolicy
from grl.rl_apps.scenarios.scenario import NFSPScenario
from grl.rl_apps.scenarios.scenario_catalog import scenario_catalog
from grl.rl_apps.scenarios.scenario_catalog.common import default_if_creating_ray_head
from grl.rl_apps.scenarios.stopping_conditions import NoStoppingCondition
from grl.rl_apps.scenarios.trainer_configs.poker_nfsp_configs import *
from grl.rllib_tools.modified_policies import SimpleQTorchPolicyPatched

scenario_catalog.add(NFSPScenario(
    name="kuhn_nfsp_dqn",
    ray_cluster_cpus=default_if_creating_ray_head(default=4),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=1,
    env_class=PokerMultiAgentEnv,
    env_config={
        "version": "kuhn_poker",
        "fixed_players": True,
    },
    trainer_class=DQNTrainer,
    avg_trainer_class=NFSPTrainer,
    policy_classes={
        "average_policy": NFSPTorchAveragePolicy,
        "best_response": SimpleQTorchPolicyPatched,
    },
    get_trainer_config=nfsp_kuhn_dqn_params,
    get_avg_trainer_config=nfsp_kuhn_avg_policy_params,
    anticipatory_param=0.1,
    nfsp_get_stopping_condition=lambda: NoStoppingCondition(),
    calculate_openspiel_metanash=True,
    calculate_openspiel_metanash_at_end=False,
    calc_metanash_every_n_iters=100,
    checkpoint_every_n_iters=None
))
