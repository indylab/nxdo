import os
import urllib.parse
from grl.rl_apps.kuhn_poker_p2sro.config import psro_leduc_dqn_params_gpu, psro_kuhn_dqn_params_gpu, psro_oshi_zumo_dqn_params_like_leduc_gpu, psro_20x_dummy_leduc_params_gpu, \
    psro_20x_dummy_leduc_params_gpu_v2, psro_12_no_limit_leduc_params_gpu, psro_30_no_limit_leduc_params_gpu, psro_60_no_limit_leduc_params_gpu, \
    _30_NO_LIMIT_LEDUC_OBS_LEN, _12_NO_LIMIT_LEDUC_OBS_LEN, _60_NO_LIMIT_LEDUC_OBS_LEN, psro_40x_dummy_leduc_params_gpu, psro_80x_dummy_leduc_params_gpu


from grl.rl_apps.nfsp.config import nfsp_kuhn_dqn_params_gpu, nfsp_kuhn_avg_policy_params_gpu, \
    nfsp_leduc_avg_policy_params_gpu, nfsp_leduc_dqn_params_gpu, nfsp_oshi_zumo_avg_policy_params_like_leduc_gpu, \
    nfsp_oshi_zumo_dqn_params_like_leduc_gpu, nfsp_20x_dummy_leduc_params_gpu, nfsp_20x_dummy_leduc_avg_policy_params_gpu, \
    nfsp_20x_dummy_leduc_avg_policy_params_gpu_v2, nfsp_20x_dummy_leduc_params_gpu_v2, nfsp_12_no_limit_leduc_avg_policy_params_gpu, \
    nfsp_12_no_limit_leduc_params_gpu, nfsp_30_no_limit_leduc_params_gpu, nfsp_30_no_limit_leduc_avg_policy_params_gpu, \
    nfsp_60_no_limit_leduc_params_gpu, nfsp_60_no_limit_leduc_avg_policy_params_gpu, nfsp_40x_dummy_leduc_params_gpu, \
    nfsp_40x_dummy_leduc_avg_policy_params_gpu, nfsp_80x_dummy_leduc_params_gpu, nfsp_80x_dummy_leduc_avg_policy_params_gpu

from grl.rl_apps.kuhn_poker_p2sro.poker_multi_agent_env import PokerMultiAgentEnv
from grl.rl_apps.kuhn_poker_p2sro.oshi_zumo_multi_agent_env import OshiZumoMultiAgentEnv
from grl.rl_apps.scenarios.stopping_conditions import EpisodesSingleBRRewardPlateauStoppingCondition, NoStoppingCondition, \
    TimeStepsSingleBRRewardPlateauStoppingCondition, StopImmediately

from ray.rllib.agents.dqn import DQNTrainer


from grl.nfsp_rllib.nfsp import NFSPTrainer, NFSPTorchAveragePolicy
from grl.rllib_tools.modified_policies import SimpleQTorchPolicyPatched

from grl.rl_apps.xfdo.solve_restricted_game_fns import SolveRestrictedGameFixedRewardThreshold, SolveRestrictedGameDynamicRewardThreshold1

from grl.rllib_tools.valid_actions_fcnet import get_valid_action_fcn_class
from grl.rl_apps.kuhn_poker_p2sro.poker_multi_agent_env import OBS_SHAPES, LEDUC_POKER

_LEDUC_OBS_LEN = OBS_SHAPES[LEDUC_POKER][0]

_GRL_SEED = int(os.getenv("GRL_SEED", 0))
_RAY_ADDRESS = os.getenv("RAY_ADDRESS")
_CREATE_RAY_HEAD = _RAY_ADDRESS is not None and len(_RAY_ADDRESS) > 0


def _ray_port_with_default_and_seed(default_port: int) -> int:
    if _RAY_ADDRESS:
        return urllib.parse.urlsplit('//' + _RAY_ADDRESS).port
    return default_port + _GRL_SEED


def _default_if_creating_ray_head(default):
    if _CREATE_RAY_HEAD:
        return None
    return default


scenarios = {

    # NFSP #########################################################

    # "kuhn_nfsp_dqn": {
    #     "env_class": PokerMultiAgentEnv,
    #     "env_config": {
    #         'version': "kuhn_poker",
    #         "fixed_players": True,
    #     },
    #     "trainer_class": DQNTrainer,
    #     "avg_trainer_class": NFSPTrainer,
    #     "policy_classes": {
    #         "average_policy": NFSPTorchAveragePolicy,
    #         "best_response": SimpleQTorchPolicyPatched,
    #     },
    #     "get_trainer_config": nfsp_kuhn_dqn_params,
    #     "get_avg_trainer_config": nfsp_kuhn_avg_policy_params,
    #     "anticipatory_param": 0.1,
    #     "nfsp_get_stopping_condition": lambda: NoStoppingCondition(),
    #     "calculate_openspiel_metanash": True,
    #    "calculate_openspiel_metanash_at_end": False,
    #     "calc_metanash_every_n_iters": 500,
    # },

    "kuhn_nfsp_dqn_gpu": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=4),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,

        
        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "kuhn_poker",
            "fixed_players": True,
        },
        "trainer_class": DQNTrainer,
        "avg_trainer_class": NFSPTrainer,
        "policy_classes": {
            "average_policy": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "get_trainer_config": nfsp_kuhn_dqn_params_gpu,
        "get_avg_trainer_config": nfsp_kuhn_avg_policy_params_gpu,
        "anticipatory_param": 0.1,
        "nfsp_get_stopping_condition": lambda: NoStoppingCondition(),
        "calculate_openspiel_metanash": True,
        "calculate_openspiel_metanash_at_end": False,
        "calc_metanash_every_n_iters": 100,
        "checkpoint_every_n_iters": None,
    },

    "leduc_nfsp_dqn_gpu": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=4),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,


        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "leduc_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
        },
        "trainer_class": DQNTrainer,
        "avg_trainer_class": NFSPTrainer,
        "policy_classes": {
            "average_policy": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "get_trainer_config": nfsp_leduc_dqn_params_gpu,
        "get_avg_trainer_config": nfsp_leduc_avg_policy_params_gpu,
        "anticipatory_param": 0.1,
        "nfsp_get_stopping_condition": lambda: NoStoppingCondition(),
        "calculate_openspiel_metanash": True,
        "calculate_openspiel_metanash_at_end": False,
        "calc_metanash_every_n_iters": 100,
        "checkpoint_every_n_iters": None,
    },

    "12_no_limit_leduc_nfsp_dqn_gpu": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=4),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,


        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "universal_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "universal_poker_stack_size": 12,
        },
        "trainer_class": DQNTrainer,
        "avg_trainer_class": NFSPTrainer,
        "policy_classes": {
            "average_policy": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "get_trainer_config": nfsp_12_no_limit_leduc_params_gpu,
        "get_avg_trainer_config": nfsp_12_no_limit_leduc_avg_policy_params_gpu,
        "anticipatory_param": 0.1,
        "nfsp_get_stopping_condition": lambda: NoStoppingCondition(),
        "calculate_openspiel_metanash": False,
        "calculate_openspiel_metanash_at_end": False,
        "calc_metanash_every_n_iters": 100,
        "checkpoint_every_n_iters": 400,
    },

    "30_no_limit_leduc_nfsp_dqn_gpu": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=4),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,


        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "universal_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "universal_poker_stack_size": 30,
        },
        "trainer_class": DQNTrainer,
        "avg_trainer_class": NFSPTrainer,
        "policy_classes": {
            "average_policy": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "get_trainer_config": nfsp_30_no_limit_leduc_params_gpu,
        "get_avg_trainer_config": nfsp_30_no_limit_leduc_avg_policy_params_gpu,
        "anticipatory_param": 0.1,
        "nfsp_get_stopping_condition": lambda: NoStoppingCondition(),
        "calculate_openspiel_metanash": False,
        "calculate_openspiel_metanash_at_end": False,
        "calc_metanash_every_n_iters": 100,
        "checkpoint_every_n_iters": 400,
    },


    "60_no_limit_leduc_nfsp_dqn_gpu": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=4),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,


        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "universal_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "universal_poker_stack_size": 60,
        },
        "trainer_class": DQNTrainer,
        "avg_trainer_class": NFSPTrainer,
        "policy_classes": {
            "average_policy": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "get_trainer_config": nfsp_60_no_limit_leduc_params_gpu,
        "get_avg_trainer_config": nfsp_60_no_limit_leduc_avg_policy_params_gpu,
        "anticipatory_param": 0.1,
        "nfsp_get_stopping_condition": lambda: NoStoppingCondition(),
        "calculate_openspiel_metanash": False,
        "calculate_openspiel_metanash_at_end": False,
        "calc_metanash_every_n_iters": 100,
        "checkpoint_every_n_iters": 400,
    },

    "20x_dummy_leduc_nfsp_dqn_gpu": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=4),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,


        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "leduc_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "dummy_action_multiplier": 20,
        },
        "trainer_class": DQNTrainer,
        "avg_trainer_class": NFSPTrainer,
        "policy_classes": {
            "average_policy": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "get_trainer_config": nfsp_20x_dummy_leduc_params_gpu,
        "get_avg_trainer_config": nfsp_20x_dummy_leduc_avg_policy_params_gpu,
        "anticipatory_param": 0.1,
        "nfsp_get_stopping_condition": lambda: NoStoppingCondition(),
        "calculate_openspiel_metanash": True,
        "calculate_openspiel_metanash_at_end": False,
        "calc_metanash_every_n_iters": 100,
        "checkpoint_every_n_iters": None,
    },

    "40x_dummy_leduc_nfsp_dqn_gpu": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=4),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,


        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "leduc_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "dummy_action_multiplier": 40,
        },
        "trainer_class": DQNTrainer,
        "avg_trainer_class": NFSPTrainer,
        "policy_classes": {
            "average_policy": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "get_trainer_config": nfsp_40x_dummy_leduc_params_gpu,
        "get_avg_trainer_config": nfsp_40x_dummy_leduc_avg_policy_params_gpu,
        "anticipatory_param": 0.1,
        "nfsp_get_stopping_condition": lambda: NoStoppingCondition(),
        "calculate_openspiel_metanash": True,
        "calculate_openspiel_metanash_at_end": False,
        "calc_metanash_every_n_iters": 100,
        "checkpoint_every_n_iters": None,
    },


    "80x_dummy_leduc_nfsp_dqn_gpu": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=4),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,


        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "leduc_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "dummy_action_multiplier": 80,
        },
        "trainer_class": DQNTrainer,
        "avg_trainer_class": NFSPTrainer,
        "policy_classes": {
            "average_policy": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "get_trainer_config": nfsp_80x_dummy_leduc_params_gpu,
        "get_avg_trainer_config": nfsp_80x_dummy_leduc_avg_policy_params_gpu,
        "anticipatory_param": 0.1,
        "nfsp_get_stopping_condition": lambda: NoStoppingCondition(),
        "calculate_openspiel_metanash": True,
        "calculate_openspiel_metanash_at_end": False,
        "calc_metanash_every_n_iters": 100,
        "checkpoint_every_n_iters": None,
    },


    "20x_dummy_leduc_nfsp_dqn_gpu_v2": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=4),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,


        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "leduc_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "dummy_action_multiplier": 20,
        },
        "trainer_class": DQNTrainer,
        "avg_trainer_class": NFSPTrainer,
        "policy_classes": {
            "average_policy": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "get_trainer_config": nfsp_20x_dummy_leduc_params_gpu_v2,
        "get_avg_trainer_config": nfsp_20x_dummy_leduc_avg_policy_params_gpu_v2,
        "anticipatory_param": 0.1,
        "nfsp_get_stopping_condition": lambda: NoStoppingCondition(),
        "calculate_openspiel_metanash": True,
        "calculate_openspiel_metanash_at_end": False,
        "calc_metanash_every_n_iters": 100,
        "checkpoint_every_n_iters": None,
    },

    "oshi_zumo_nfsp_dqn_gpu": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=4),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,

        "env_class": OshiZumoMultiAgentEnv,
        "env_config": {
            'version': "oshi_zumo",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
        },
        "trainer_class": DQNTrainer,
        "avg_trainer_class": NFSPTrainer,
        "policy_classes": {
            "average_policy": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "get_trainer_config": nfsp_oshi_zumo_dqn_params_like_leduc_gpu,
        "get_avg_trainer_config": nfsp_oshi_zumo_avg_policy_params_like_leduc_gpu,
        "anticipatory_param": 0.1,
        "nfsp_get_stopping_condition": lambda: NoStoppingCondition(),
        "calculate_openspiel_metanash": False,
        "calculate_openspiel_metanash_at_end": False,
        "calc_metanash_every_n_iters": 100,
        "checkpoint_every_n_iters": 400,
    },


    #
    # "20x_dummy_kuhn_nfsp_dqn": {
    #     "env_class": PokerMultiAgentEnv,
    #     "env_config": {
    #         'version': "kuhn_poker",
    #         "fixed_players": True,
    #         "dummy_action_multiplier": 20,
    #     },
    #     "trainer_class": DQNTrainer,
    #     "avg_trainer_class": NFSPTrainer,
    #     "policy_classes": {
    #         "average_policy": NFSPTorchAveragePolicy,
    #         "best_response": SimpleQTorchPolicyPatched,
    #     },
    #     "get_trainer_config": nfsp_kuhn_dqn_params,
    #     "get_avg_trainer_config": nfsp_kuhn_avg_policy_params,
    #     "anticipatory_param": 0.1,
    #     "nfsp_get_stopping_condition": lambda: NoStoppingCondition(),
    #     "calculate_openspiel_metanash": True,
    #    "calculate_openspiel_metanash_at_end": False,
    #     "calc_metanash_every_n_iters": 500,
    # },
    #
    # "leduc_nfsp_dqn": {
    #     "env_class": PokerMultiAgentEnv,
    #     "env_config": {
    #         'version': "leduc_poker",
    #         "fixed_players": True,
    #         "append_valid_actions_mask_to_obs": True,
    #     },
    #     "trainer_class": DQNTrainer,
    #     "avg_trainer_class": NFSPTrainer,
    #     "policy_classes": {
    #         "average_policy": NFSPTorchAveragePolicy,
    #         "best_response": SimpleQTorchPolicyPatched,
    #     },
    #     "get_trainer_config": nfsp_leduc_dqn_params,
    #     "get_avg_trainer_config": nfsp_leduc_avg_policy_params,
    #     "anticipatory_param": 0.1,
    #     "nfsp_get_stopping_condition": lambda: NoStoppingCondition(),
    #     "calculate_openspiel_metanash": True,
    #    "calculate_openspiel_metanash_at_end": False,
    #     "calc_metanash_every_n_iters": 2000,
    # },
    #


    #
    # "oshi_zumo_nfsp_dqn_kuhn_params": {
    #     "env_class": OshiZumoMultiAgentEnv,
    #     "env_config": {
    #         'version': "oshi_zumo",
    #         "fixed_players": True,
    #         "append_valid_actions_mask_to_obs": True,
    #     },
    #     "trainer_class": DQNTrainer,
    #     "avg_trainer_class": NFSPTrainer,
    #     "policy_classes": {
    #         "average_policy": NFSPTorchAveragePolicy,
    #         "best_response": SimpleQTorchPolicyPatched,
    #     },
    #     "get_trainer_config": nfsp_oshi_zumo_dqn_params_like_kuhn,
    #     "anticipatory_param": 0.1,
    #     "nfsp_get_stopping_condition": lambda: NoStoppingCondition(),
    #     "calculate_openspiel_metanash": True,
    #    "calculate_openspiel_metanash_at_end": False,
    #     "calc_metanash_every_n_iters": 2000,
    # },

    # CFP #########################################################

    # "kuhn_cfp_dqn": {
    #     "env_class": PokerMultiAgentEnv,
    #     "env_config": {
    #         'version': "kuhn_poker",
    #         "fixed_players": True,
    #     },
    #     "trainer_class": DQNTrainer,
    #     "policy_classes": {
    #         "best_response": SimpleQTorchPolicyWithActionProbsOut,
    #     },
    #     "get_trainer_config": nfsp_kuhn_dqn_params,
    #     "anticipatory_param": 0.1,
    #     "checkpoint_reservoir_size": 5000,
    #     "cfp_get_stopping_condition": lambda: NoStoppingCondition(),
    #     "calculate_openspiel_metanash": True,
    #    "calculate_openspiel_metanash_at_end": False,
    #     "calc_metanash_every_n_iters": 500,
    # },
    #
    # "20x_dummy_kuhn_cfp_dqn": {
    #     "env_class": PokerMultiAgentEnv,
    #     "env_config": {
    #         'version': "kuhn_poker",
    #         "fixed_players": True,
    #         "dummy_action_multiplier": 20,
    #     },
    #     "trainer_class": DQNTrainer,
    #     "policy_classes": {
    #         "best_response": SimpleQTorchPolicyWithActionProbsOut,
    #     },
    #     "get_trainer_config": nfsp_kuhn_dqn_params,
    #     "anticipatory_param": 0.1,
    #     "checkpoint_reservoir_size": 5000,
    #     "cfp_get_stopping_condition": lambda: NoStoppingCondition(),
    #     "calculate_openspiel_metanash": True,
    #    "calculate_openspiel_metanash_at_end": False,
    #     "calc_metanash_every_n_iters": 500,
    # },
    #
    # "leduc_cfp_dqn": {
    #     "env_class": PokerMultiAgentEnv,
    #     "env_config": {
    #         'version': "leduc_poker",
    #         "fixed_players": True,
    #         "append_valid_actions_mask_to_obs": True,
    #     },
    #     "trainer_class": DQNTrainer,
    #     "policy_classes": {
    #         "best_response": SimpleQTorchPolicyWithActionProbsOut,
    #     },
    #     "get_trainer_config": nfsp_leduc_dqn_params,
    #     "anticipatory_param": 0.1,
    #     "checkpoint_reservoir_size": 5000,
    #     "cfp_get_stopping_condition": lambda: NoStoppingCondition(),
    #     "calculate_openspiel_metanash": True,
    #    "calculate_openspiel_metanash_at_end": False,
    #     "calc_metanash_every_n_iters": 2000,
    # },
    #
    # "20x_dummy_leduc_cfp_dqn": {
    #     "env_class": PokerMultiAgentEnv,
    #     "env_config": {
    #         'version': "leduc_poker",
    #         "fixed_players": True,
    #         "append_valid_actions_mask_to_obs": True,
    #         "dummy_action_multiplier": 20,
    #     },
    #     "trainer_class": DQNTrainer,
    #     "policy_classes": {
    #         "best_response": SimpleQTorchPolicyWithActionProbsOut,
    #     },
    #     "get_trainer_config": nfsp_leduc_dqn_params,
    #     "anticipatory_param": 0.1,
    #     "checkpoint_reservoir_size": 5000,
    #     "cfp_get_stopping_condition": lambda: NoStoppingCondition(),
    #     "calculate_openspiel_metanash": True,
    #    "calculate_openspiel_metanash_at_end": False,
    #     "calc_metanash_every_n_iters": 2000,
    # },

    # PSRO #########################################################

    # If doing multiple seeds on the same machine,
    # make a new scenario for each seed with different ports in a for-loop.

    # "kuhn_psro_dqn": {
    #     "env_class": PokerMultiAgentEnv,
    #     "env_config": {
    #         'version': "kuhn_poker",
    #         "fixed_players": True,
    #     },
    #     "trainer_class": DQNTrainer,
    #     "policy_classes": {
    #         "metanash": SimpleQTorchPolicyPatched,
    #         "best_response": SimpleQTorchPolicyPatched,
    #         "eval": SimpleQTorchPolicyPatched,
    #     },
    #     "psro_port": 4100,
    #     "eval_port": 4200,
    #     "num_eval_workers": 8,
    #     "games_per_payoff_eval": 20000,
    #     "p2sro": False,
    #     "get_trainer_config": psro_kuhn_dqn_params,
    #     "psro_get_stopping_condition": lambda: SingleBRRewardPlateauStoppingCondition(
    #         br_policy_id="best_response",
    #         dont_check_plateau_before_n_iters=300,
    #         check_plateau_every_n_iters=100,
    #         minimum_reward_improvement_otherwise_plateaued=0.01,
    #         max_train_iters=20000,
    #     ),
    # },
    #
    # "20x_dummy_kuhn_psro_dqn": {
    #     "env_class": PokerMultiAgentEnv,
    #     "env_config": {
    #         'version': "kuhn_poker",
    #         "fixed_players": True,
    #         "dummy_action_multiplier": 20,
    #     },
    #     "trainer_class": DQNTrainer,
    #     "policy_classes": {
    #         "metanash": SimpleQTorchPolicyPatched,
    #         "best_response": SimpleQTorchPolicyPatched,
    #         "eval": SimpleQTorchPolicyPatched,
    #     },
    #     "psro_port": 4105,
    #     "eval_port": 4205,
    #     "num_eval_workers": 8,
    #     "games_per_payoff_eval": 20000,
    #     "p2sro": False,
    #     "get_trainer_config": psro_kuhn_dqn_params,
    #     "psro_get_stopping_condition": lambda: SingleBRRewardPlateauStoppingCondition(
    #         br_policy_id="best_response",
    #         dont_check_plateau_before_n_iters=300,
    #         check_plateau_every_n_iters=100,
    #         minimum_reward_improvement_otherwise_plateaued=0.01,
    #         max_train_iters=20000,
    #     ),
    # },
    #
    # "leduc_psro_dqn": {
    #     "env_class": PokerMultiAgentEnv,
    #     "env_config": {
    #         'version': "leduc_poker",
    #         "fixed_players": True,
    #         "append_valid_actions_mask_to_obs": True,
    #     },
    #     "trainer_class": DQNTrainer,
    #     "policy_classes": {
    #         "metanash": SimpleQTorchPolicyPatched,
    #         "best_response": SimpleQTorchPolicyPatched,
    #         "eval": SimpleQTorchPolicyPatched,
    #     },
    #     "psro_port": 4110,
    #     "eval_port": 4210,
    #     "num_eval_workers": 8,
    #     "games_per_payoff_eval": 20000,
    #     "p2sro": False,
    #     "get_trainer_config": psro_leduc_dqn_params,
    #     "psro_get_stopping_condition": lambda: SingleBRRewardPlateauStoppingCondition(
    #         br_policy_id="best_response",
    #         dont_check_plateau_before_n_iters=3000,
    #         check_plateau_every_n_iters=1000,
    #         minimum_reward_improvement_otherwise_plateaued=0.01,
    #         max_train_iters=20000,
    #     ),
    # },

    "kuhn_psro_dqn_gpu": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,


        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "kuhn_poker",
            "fixed_players": True,
        },
        "mix_metanash_with_uniform_dist_coeff": 0.0,
        "trainer_class": DQNTrainer,
        "policy_classes": {
            "metanash": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
            "eval": SimpleQTorchPolicyPatched,
        },
        "psro_port": 4100 + _GRL_SEED,
        "eval_port": 4200 + _GRL_SEED,
        "num_eval_workers": 8,
        "games_per_payoff_eval": 20000,
        "p2sro": False,
        "get_trainer_config": psro_kuhn_dqn_params_gpu,
        "psro_get_stopping_condition": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),
    },


    "leduc_psro_dqn_gpu": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,


        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "leduc_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
        },
        "mix_metanash_with_uniform_dist_coeff": 0.0,
        "trainer_class": DQNTrainer,
        "policy_classes": {
            "metanash": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
            "eval": SimpleQTorchPolicyPatched,
        },
        "psro_port": 4105 + _GRL_SEED,
        "eval_port": 4205 + _GRL_SEED,
        "num_eval_workers": 8,
        "games_per_payoff_eval": 3000,
        "p2sro": False,
        "get_trainer_config": psro_leduc_dqn_params_gpu,
        "psro_get_stopping_condition": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),
    },


    "20x_dummy_leduc_psro_dqn_gpu": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,


        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "leduc_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "dummy_action_multiplier": 20,
        },
        "mix_metanash_with_uniform_dist_coeff": 0.0,
        "trainer_class": DQNTrainer,
        "policy_classes": {
            "metanash": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
            "eval": SimpleQTorchPolicyPatched,
        },
        "psro_port": 4110 + _GRL_SEED,
        "eval_port": 4210 + _GRL_SEED,
        "num_eval_workers": 8,
        "games_per_payoff_eval": 3000,
        "p2sro": False,
        "get_trainer_config": psro_20x_dummy_leduc_params_gpu,
        "psro_get_stopping_condition": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),
    },

    "20x_dummy_leduc_psro_dqn_gpu_slow_stop": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,

        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "leduc_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "dummy_action_multiplier": 20,
        },
        "mix_metanash_with_uniform_dist_coeff": 0.0,
        "trainer_class": DQNTrainer,
        "policy_classes": {
            "metanash": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
            "eval": SimpleQTorchPolicyPatched,
        },
        "psro_port": 4125 + _GRL_SEED,
        "eval_port": 4225 + _GRL_SEED,
        "num_eval_workers": 8,
        "games_per_payoff_eval": 3000,
        "p2sro": False,
        "get_trainer_config": psro_20x_dummy_leduc_params_gpu,
        "psro_get_stopping_condition": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(4e4),
            check_plateau_every_n_episodes=int(4e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(2e5),
        ),
    },

    "12_no_limit_leduc_psro_dqn_gpu": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,


        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "universal_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "universal_poker_stack_size": 12,
        },
        "mix_metanash_with_uniform_dist_coeff": 0.0,
        "trainer_class": DQNTrainer,
        "policy_classes": {
            "metanash": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
            "eval": SimpleQTorchPolicyPatched,
        },
        "psro_port": 4130 + _GRL_SEED,
        "eval_port": 4230 + _GRL_SEED,
        "num_eval_workers": 8,
        "games_per_payoff_eval": 4000,
        "p2sro": False,
        "get_trainer_config": psro_12_no_limit_leduc_params_gpu,
        "psro_get_stopping_condition": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),
    },

    "30_no_limit_leduc_psro_dqn_gpu": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,


        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "universal_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "universal_poker_stack_size": 30,
        },
        "mix_metanash_with_uniform_dist_coeff": 0.0,
        "trainer_class": DQNTrainer,
        "policy_classes": {
            "metanash": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
            "eval": SimpleQTorchPolicyPatched,
        },
        "psro_port": 4135 + _GRL_SEED,
        "eval_port": 4235 + _GRL_SEED,
        "num_eval_workers": 8,
        "games_per_payoff_eval": 4000,
        "p2sro": False,
        "get_trainer_config": psro_30_no_limit_leduc_params_gpu,
        "psro_get_stopping_condition": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),
    },

    "60_no_limit_leduc_psro_dqn_gpu": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,


        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "universal_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "universal_poker_stack_size": 60,
        },
        "mix_metanash_with_uniform_dist_coeff": 0.0,
        "trainer_class": DQNTrainer,
        "policy_classes": {
            "metanash": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
            "eval": SimpleQTorchPolicyPatched,
        },
        "psro_port": 4140 + _GRL_SEED,
        "eval_port": 4240 + _GRL_SEED,
        "num_eval_workers": 8,
        "games_per_payoff_eval": 4000,
        "p2sro": False,
        "get_trainer_config": psro_12_no_limit_leduc_params_gpu,
        "psro_get_stopping_condition": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),
    },

    "40x_dummy_leduc_psro_dqn_gpu": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,


        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "leduc_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "dummy_action_multiplier": 40,
        },
        "mix_metanash_with_uniform_dist_coeff": 0.0,
        "trainer_class": DQNTrainer,
        "policy_classes": {
            "metanash": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
            "eval": SimpleQTorchPolicyPatched,
        },
        "psro_port": 4145 + _GRL_SEED,
        "eval_port": 4245 + _GRL_SEED,
        "num_eval_workers": 8,
        "games_per_payoff_eval": 3000,
        "p2sro": False,
        "get_trainer_config": psro_40x_dummy_leduc_params_gpu,
        "psro_get_stopping_condition": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),
    },

    "oshi_zumo_psro_dqn_gpu": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,

        "env_class": OshiZumoMultiAgentEnv,
        "env_config": {
            'version': "oshi_zumo",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
        },
        "mix_metanash_with_uniform_dist_coeff": 0.0,
        "trainer_class": DQNTrainer,
        "policy_classes": {
            "metanash": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
            "eval": SimpleQTorchPolicyPatched,
        },
        "psro_port": 4150 + _GRL_SEED,
        "eval_port": 4250 + _GRL_SEED,
        "num_eval_workers": 8,
        "games_per_payoff_eval": 2000,
        "p2sro": False,
        "get_trainer_config": psro_oshi_zumo_dqn_params_like_leduc_gpu,
        "psro_get_stopping_condition": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),
    },


    "80x_dummy_leduc_psro_dqn_gpu": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,


        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "leduc_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "dummy_action_multiplier": 80,
        },
        "mix_metanash_with_uniform_dist_coeff": 0.0,
        "trainer_class": DQNTrainer,
        "policy_classes": {
            "metanash": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
            "eval": SimpleQTorchPolicyPatched,
        },
        "psro_port": 4155 + _GRL_SEED,
        "eval_port": 4255 + _GRL_SEED,
        "num_eval_workers": 8,
        "games_per_payoff_eval": 3000,
        "p2sro": False,
        "get_trainer_config": psro_80x_dummy_leduc_params_gpu,
        "psro_get_stopping_condition": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),
    },

    # "20x_dummy_leduc_psro_dqn_gpu_v2": {
    #     "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
    #     "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
    #     "ray_object_store_memory_cap_gigabytes": 1,

    #
    #     "env_class": PokerMultiAgentEnv,
    #     "env_config": {
    #         'version': "leduc_poker",
    #         "fixed_players": True,
    #         "append_valid_actions_mask_to_obs": True,
    #         "dummy_action_multiplier": 20,
    #     },
    #     "mix_metanash_with_uniform_dist_coeff": 0.0,
    #     "trainer_class": DQNTrainer,
    #     "policy_classes": {
    #         "metanash": SimpleQTorchPolicyPatched,
    #         "best_response": SimpleQTorchPolicyPatched,
    #         "eval": SimpleQTorchPolicyPatched,
    #     },
    #     "psro_port": 4120 + _GRL_SEED,
    #     "eval_port": 4220 + _GRL_SEED,
    #     "num_eval_workers": 8,
    #     "games_per_payoff_eval": 3000,
    #     "p2sro": False,
    #     "get_trainer_config": psro_20x_dummy_leduc_params_gpu_v2,
    #     "psro_get_stopping_condition": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
    #         br_policy_id="best_response",
    #         dont_check_plateau_before_n_episodes=int(4e4),
    #         check_plateau_every_n_episodes=int(4e4),
    #         minimum_reward_improvement_otherwise_plateaued=0.01,
    #         max_train_episodes=int(2e5),
    #     ),
    # },




    #
    # "oshi_zumo_psro_dqn_kuhn_params": {
    #     "env_class": OshiZumoMultiAgentEnv,
    #     "env_config": {
    #         'version': "oshi_zumo",
    #         "fixed_players": True,
    #         "append_valid_actions_mask_to_obs": True,
    #     },
    #     "trainer_class": DQNTrainer,
    #     "policy_classes": {
    #         "metanash": SimpleQTorchPolicyPatched,
    #         "best_response": SimpleQTorchPolicyPatched,
    #         "eval": SimpleQTorchPolicyPatched,
    #     },
    #     "psro_port": 4125,
    #     "eval_port": 4225,
    #     "num_eval_workers": 8,
    #     "games_per_payoff_eval": 20000,
    #     "p2sro": False,
    #     "get_trainer_config": nfsp_oshi_zumo_dqn_params_like_kuhn,
    #     "psro_get_stopping_condition": lambda: SingleBRRewardPlateauStoppingCondition(
    #         br_policy_id="best_response",
    #         dont_check_plateau_before_n_iters=300,
    #         check_plateau_every_n_iters=100,
    #         minimum_reward_improvement_otherwise_plateaued=0.01,
    #         max_train_iters=20000,
    #     ),
    # },

    # XFDO ############################################################################################

    "kuhn_xfdo_dqn_nfsp_gpu": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,


        "xfdo_port": 4400 + _GRL_SEED,
        "use_openspiel_restricted_game": False,
        "restricted_game_custom_model": None,
        "xfdo_metanash_method": "nfsp",
        "get_restricted_game_solver": lambda scenario: SolveRestrictedGameFixedRewardThreshold(
            scenario=scenario, br_reward_threshold=0.01, min_episodes=300000,
            required_fields=["z_avg_policy_exploitability"]
        ),

        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "kuhn_poker",
            "fixed_players": True,
        },

        "trainer_class_br": DQNTrainer,
        "policy_classes_br": {
            "metanash": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },

        "get_trainer_config_br": psro_kuhn_dqn_params_gpu,
        "get_stopping_condition_br": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),

        "trainer_class_nfsp": DQNTrainer,
        "avg_trainer_class_nfsp": NFSPTrainer,
        "policy_classes_nfsp": {
            "average_policy": NFSPTorchAveragePolicy,
            "delegate_policy": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "anticipatory_param_nfsp": 0.1,
        "get_trainer_config_nfsp": nfsp_kuhn_dqn_params_gpu,
        "get_avg_trainer_config_nfsp": nfsp_kuhn_avg_policy_params_gpu,
        "calculate_openspiel_metanash": True,
        "calculate_openspiel_metanash_at_end": False,
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },

    "leduc_xfdo_dqn_nfsp_gpu": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,


        "xfdo_port": 4405 + _GRL_SEED,
        "use_openspiel_restricted_game": False,
        "restricted_game_custom_model": None,
        "xfdo_metanash_method": "nfsp",
        "get_restricted_game_solver": lambda scenario: SolveRestrictedGameFixedRewardThreshold(
            scenario=scenario, br_reward_threshold=0.1, min_episodes=300000,
            required_fields=["z_avg_policy_exploitability"]
        ),

        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "leduc_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
        },

        "trainer_class_br": DQNTrainer,
        "policy_classes_br": {
            "metanash": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },

        "get_trainer_config_br": psro_leduc_dqn_params_gpu,
        "get_stopping_condition_br": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),

        "trainer_class_nfsp": DQNTrainer,
        "avg_trainer_class_nfsp": NFSPTrainer,
        "policy_classes_nfsp": {
            "average_policy": NFSPTorchAveragePolicy,
            "delegate_policy": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "anticipatory_param_nfsp": 0.1,
        "get_trainer_config_nfsp": nfsp_leduc_dqn_params_gpu,
        "get_avg_trainer_config_nfsp": nfsp_leduc_avg_policy_params_gpu,
        "calculate_openspiel_metanash": True,
        "calculate_openspiel_metanash_at_end": False,
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },

    "20x_dummy_leduc_xfdo_dqn_nfsp_gpu": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,


        "xfdo_port": 4410 + _GRL_SEED,
        "use_openspiel_restricted_game": False,
        "restricted_game_custom_model": None,
        "xfdo_metanash_method": "nfsp",
        "get_restricted_game_solver": lambda scenario: SolveRestrictedGameFixedRewardThreshold(
            scenario=scenario, br_reward_threshold=0.1, min_episodes=300000,
            required_fields=["z_avg_policy_exploitability"]
        ),

        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "leduc_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "dummy_action_multiplier": 20,
        },

        "trainer_class_br": DQNTrainer,
        "policy_classes_br": {
            "metanash": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },

        "get_trainer_config_br": psro_20x_dummy_leduc_params_gpu,
        "get_stopping_condition_br": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),

        "trainer_class_nfsp": DQNTrainer,
        "avg_trainer_class_nfsp": NFSPTrainer,
        "policy_classes_nfsp": {
            "average_policy": NFSPTorchAveragePolicy,
            "delegate_policy": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "anticipatory_param_nfsp": 0.1,
        "get_trainer_config_nfsp": nfsp_20x_dummy_leduc_params_gpu,
        "get_avg_trainer_config_nfsp": nfsp_20x_dummy_leduc_avg_policy_params_gpu,
        "calculate_openspiel_metanash": True,
        "calculate_openspiel_metanash_at_end": False,
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },

    "20x_dummy_leduc_xfdo_dqn_nfsp_gpu_dynamic_threshold_1": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,


        "xfdo_port": 4415 + _GRL_SEED,
        "use_openspiel_restricted_game": False,
        "restricted_game_custom_model": None,
        "xfdo_metanash_method": "nfsp",
        "get_restricted_game_solver": lambda scenario: SolveRestrictedGameDynamicRewardThreshold1(
            scenario=scenario,
            dont_solve_first_n_xfdo_iters=5,
            starting_rew_threshold=2.0,
            min_rew_threshold=0.1,
            min_episodes=50000,
            epsilon=0.05,
            required_fields=["z_avg_policy_exploitability"],
        ),

        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "leduc_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "dummy_action_multiplier": 20,
        },

        "trainer_class_br": DQNTrainer,
        "policy_classes_br": {
            "metanash": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },

        "get_trainer_config_br": psro_20x_dummy_leduc_params_gpu,
        "get_stopping_condition_br": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),

        "trainer_class_nfsp": DQNTrainer,
        "avg_trainer_class_nfsp": NFSPTrainer,
        "policy_classes_nfsp": {
            "average_policy": NFSPTorchAveragePolicy,
            "delegate_policy": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "anticipatory_param_nfsp": 0.1,
        "get_trainer_config_nfsp": nfsp_20x_dummy_leduc_params_gpu,
        "get_avg_trainer_config_nfsp": nfsp_20x_dummy_leduc_avg_policy_params_gpu,
        "calculate_openspiel_metanash": True,
        "calculate_openspiel_metanash_at_end": False,
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },


    "20x_dummy_leduc_xfdo_dqn_nfsp_gpu_v2": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,


        "xfdo_port": 4420 + _GRL_SEED,
        "use_openspiel_restricted_game": False,
        "restricted_game_custom_model": None,
        "xfdo_metanash_method": "nfsp",
        "get_restricted_game_solver": lambda scenario: SolveRestrictedGameFixedRewardThreshold(
            scenario=scenario, br_reward_threshold=0.1, min_episodes=300000,
            required_fields=["z_avg_policy_exploitability"]
        ),

        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "leduc_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "dummy_action_multiplier": 20,
        },

        "trainer_class_br": DQNTrainer,
        "policy_classes_br": {
            "metanash": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },

        "get_trainer_config_br": psro_20x_dummy_leduc_params_gpu_v2,
        "get_stopping_condition_br": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(4e4),
            check_plateau_every_n_episodes=int(4e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(2e5),
        ),

        "trainer_class_nfsp": DQNTrainer,
        "avg_trainer_class_nfsp": NFSPTrainer,
        "policy_classes_nfsp": {
            "average_policy": NFSPTorchAveragePolicy,
            "delegate_policy": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "anticipatory_param_nfsp": 0.1,
        "get_trainer_config_nfsp": nfsp_20x_dummy_leduc_params_gpu_v2,
        "get_avg_trainer_config_nfsp": nfsp_20x_dummy_leduc_avg_policy_params_gpu_v2,
        "calculate_openspiel_metanash": True,
        "calculate_openspiel_metanash_at_end": False,
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },

    "20x_dummy_leduc_xfdo_dqn_nfsp_gpu_v2_dynamic_threshold_1": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,


        "xfdo_port": 4425 + _GRL_SEED,
        "use_openspiel_restricted_game": False,
        "restricted_game_custom_model": None,
        "xfdo_metanash_method": "nfsp",
        "get_restricted_game_solver": lambda scenario: SolveRestrictedGameDynamicRewardThreshold1(
            scenario=scenario,
            dont_solve_first_n_xfdo_iters=5,
            starting_rew_threshold=2.0,
            min_rew_threshold=0.1,
            min_episodes=50000,
            epsilon=0.05,
            required_fields=["z_avg_policy_exploitability"],
        ),

        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "leduc_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "dummy_action_multiplier": 20,
        },

        "trainer_class_br": DQNTrainer,
        "policy_classes_br": {
            "metanash": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },

        "get_trainer_config_br": psro_20x_dummy_leduc_params_gpu_v2,
        "get_stopping_condition_br": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(4e4),
            check_plateau_every_n_episodes=int(4e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(2e5),
        ),

        "trainer_class_nfsp": DQNTrainer,
        "avg_trainer_class_nfsp": NFSPTrainer,
        "policy_classes_nfsp": {
            "average_policy": NFSPTorchAveragePolicy,
            "delegate_policy": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "anticipatory_param_nfsp": 0.1,
        "get_trainer_config_nfsp": nfsp_20x_dummy_leduc_params_gpu_v2,
        "get_avg_trainer_config_nfsp": nfsp_20x_dummy_leduc_avg_policy_params_gpu_v2,
        "calculate_openspiel_metanash": True,
        "calculate_openspiel_metanash_at_end": False,
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },

    "leduc_xfdo_dqn_nfsp_gpu_dynamic_threshold_1": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,


        "xfdo_port": 4430 + _GRL_SEED,
        "use_openspiel_restricted_game": False,
        "restricted_game_custom_model": None,
        "xfdo_metanash_method": "nfsp",
        "get_restricted_game_solver": lambda scenario: SolveRestrictedGameDynamicRewardThreshold1(
            scenario=scenario,
            dont_solve_first_n_xfdo_iters=5,
            starting_rew_threshold=2.0,
            min_rew_threshold=0.1,
            min_episodes=50000,
            epsilon=0.05,
            required_fields=["z_avg_policy_exploitability"],
        ),

        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "leduc_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
        },

        "trainer_class_br": DQNTrainer,
        "policy_classes_br": {
            "metanash": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },

        "get_trainer_config_br": psro_leduc_dqn_params_gpu,
        "get_stopping_condition_br": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),

        "trainer_class_nfsp": DQNTrainer,
        "avg_trainer_class_nfsp": NFSPTrainer,
        "policy_classes_nfsp": {
            "average_policy": NFSPTorchAveragePolicy,
            "delegate_policy": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "anticipatory_param_nfsp": 0.1,
        "get_trainer_config_nfsp": nfsp_leduc_dqn_params_gpu,
        "get_avg_trainer_config_nfsp": nfsp_leduc_avg_policy_params_gpu,
        "calculate_openspiel_metanash": True,
        "calculate_openspiel_metanash_at_end": False,
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },


    "va_20x_dummy_leduc_xfdo_dqn_nfsp_gpu_dynamic_threshold_1": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,


        "xfdo_port": 4435 + _GRL_SEED,
        "use_openspiel_restricted_game": True,
        "restricted_game_custom_model": get_valid_action_fcn_class(obs_len=_LEDUC_OBS_LEN, action_space_n=3 * 20, dummy_actions_multiplier=1),
        "xfdo_metanash_method": "nfsp",
        "get_restricted_game_solver": lambda scenario: SolveRestrictedGameDynamicRewardThreshold1(
            scenario=scenario,
            dont_solve_first_n_xfdo_iters=5,
            starting_rew_threshold=2.0,
            min_rew_threshold=0.1,
            min_episodes=100000,
            epsilon=0.05,
            required_fields=["z_avg_policy_exploitability"],
        ),

        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "leduc_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "dummy_action_multiplier": 20,
        },

        "trainer_class_br": DQNTrainer,
        "policy_classes_br": {
            "metanash": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },

        "get_trainer_config_br": psro_20x_dummy_leduc_params_gpu,
        "get_stopping_condition_br": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),

        "trainer_class_nfsp": DQNTrainer,
        "avg_trainer_class_nfsp": NFSPTrainer,
        "policy_classes_nfsp": {
            "average_policy": NFSPTorchAveragePolicy,
            "delegate_policy": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "anticipatory_param_nfsp": 0.1,
        "get_trainer_config_nfsp": nfsp_20x_dummy_leduc_params_gpu,
        "get_avg_trainer_config_nfsp": nfsp_20x_dummy_leduc_avg_policy_params_gpu,
        "calculate_openspiel_metanash": True,
        "calculate_openspiel_metanash_at_end": False,
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },

    "20x_dummy_leduc_xfdo_dqn_nfsp_gpu_dynamic_threshold_1_aggressive": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,


        "xfdo_port": 4440 + _GRL_SEED,
        "use_openspiel_restricted_game": False,
        "restricted_game_custom_model": None,
        "xfdo_metanash_method": "nfsp",
        "get_restricted_game_solver": lambda scenario: SolveRestrictedGameDynamicRewardThreshold1(
            scenario=scenario,
            dont_solve_first_n_xfdo_iters=7,
            starting_rew_threshold=1.0,
            min_rew_threshold=0.05,
            min_episodes=50000,
            epsilon=0.05,
            required_fields=["z_avg_policy_exploitability"],
        ),

        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "leduc_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "dummy_action_multiplier": 20,
        },

        "trainer_class_br": DQNTrainer,
        "policy_classes_br": {
            "metanash": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },

        "get_trainer_config_br": psro_20x_dummy_leduc_params_gpu,
        "get_stopping_condition_br": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),

        "trainer_class_nfsp": DQNTrainer,
        "avg_trainer_class_nfsp": NFSPTrainer,
        "policy_classes_nfsp": {
            "average_policy": NFSPTorchAveragePolicy,
            "delegate_policy": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "anticipatory_param_nfsp": 0.1,
        "get_trainer_config_nfsp": nfsp_20x_dummy_leduc_params_gpu,
        "get_avg_trainer_config_nfsp": nfsp_20x_dummy_leduc_avg_policy_params_gpu,
        "calculate_openspiel_metanash": True,
        "calculate_openspiel_metanash_at_end": False,
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },

    "va_20x_dummy_leduc_xfdo_dqn_nfsp_gpu_dynamic_threshold_1_aggressive": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,


        "xfdo_port": 4445 + _GRL_SEED,
        "use_openspiel_restricted_game": True,
        "restricted_game_custom_model": get_valid_action_fcn_class(obs_len=_LEDUC_OBS_LEN, action_space_n=3 * 20, dummy_actions_multiplier=1),
        "xfdo_metanash_method": "nfsp",
        "get_restricted_game_solver": lambda scenario: SolveRestrictedGameDynamicRewardThreshold1(
            scenario=scenario,
            dont_solve_first_n_xfdo_iters=7,
            starting_rew_threshold=1.0,
            min_rew_threshold=0.05,
            min_episodes=100000,
            epsilon=0.05,
            required_fields=["z_avg_policy_exploitability"],
        ),

        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "leduc_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "dummy_action_multiplier": 20,
        },

        "trainer_class_br": DQNTrainer,
        "policy_classes_br": {
            "metanash": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },

        "get_trainer_config_br": psro_20x_dummy_leduc_params_gpu,
        "get_stopping_condition_br": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),

        "trainer_class_nfsp": DQNTrainer,
        "avg_trainer_class_nfsp": NFSPTrainer,
        "policy_classes_nfsp": {
            "average_policy": NFSPTorchAveragePolicy,
            "delegate_policy": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "anticipatory_param_nfsp": 0.1,
        "get_trainer_config_nfsp": nfsp_20x_dummy_leduc_params_gpu,
        "get_avg_trainer_config_nfsp": nfsp_20x_dummy_leduc_avg_policy_params_gpu,
        "calculate_openspiel_metanash": True,
        "calculate_openspiel_metanash_at_end": False,
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },

    # 40x dummy leduc ##########################################3

    "40x_dummy_leduc_xfdo_dqn_nfsp_gpu_dynamic_threshold_1_aggressive": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,

        "xfdo_port": 4450 + _GRL_SEED,
        "use_openspiel_restricted_game": False,
        "restricted_game_custom_model": None,
        "xfdo_metanash_method": "nfsp",
        "get_restricted_game_solver": lambda scenario: SolveRestrictedGameDynamicRewardThreshold1(
            scenario=scenario,
            dont_solve_first_n_xfdo_iters=7,
            starting_rew_threshold=1.0,
            min_rew_threshold=0.05,
            min_episodes=200000,
            epsilon=0.05,
            required_fields=["z_avg_policy_exploitability"],
        ),

        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "leduc_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "dummy_action_multiplier": 40,
        },

        "trainer_class_br": DQNTrainer,
        "policy_classes_br": {
            "metanash": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },

        "get_trainer_config_br": psro_40x_dummy_leduc_params_gpu,
        "get_stopping_condition_br": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),

        "trainer_class_nfsp": DQNTrainer,
        "avg_trainer_class_nfsp": NFSPTrainer,
        "policy_classes_nfsp": {
            "average_policy": NFSPTorchAveragePolicy,
            "delegate_policy": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "anticipatory_param_nfsp": 0.1,
        "get_trainer_config_nfsp": nfsp_40x_dummy_leduc_params_gpu,
        "get_avg_trainer_config_nfsp": nfsp_40x_dummy_leduc_avg_policy_params_gpu,
        "calculate_openspiel_metanash": True,
        "calculate_openspiel_metanash_at_end": False,
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },

    "va_40x_dummy_leduc_xfdo_dqn_nfsp_gpu_dynamic_threshold_1_aggressive": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,

        "xfdo_port": 4455 + _GRL_SEED,
        "use_openspiel_restricted_game": True,
        "restricted_game_custom_model": get_valid_action_fcn_class(obs_len=_LEDUC_OBS_LEN, action_space_n=3 * 40,
                                                                   dummy_actions_multiplier=1),
        "xfdo_metanash_method": "nfsp",
        "get_restricted_game_solver": lambda scenario: SolveRestrictedGameDynamicRewardThreshold1(
            scenario=scenario,
            dont_solve_first_n_xfdo_iters=7,
            starting_rew_threshold=1.0,
            min_rew_threshold=0.05,
            min_episodes=200000,
            epsilon=0.05,
            required_fields=["z_avg_policy_exploitability"],
        ),

        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "leduc_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "dummy_action_multiplier": 40,
        },

        "trainer_class_br": DQNTrainer,
        "policy_classes_br": {
            "metanash": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },

        "get_trainer_config_br": psro_40x_dummy_leduc_params_gpu,
        "get_stopping_condition_br": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),

        "trainer_class_nfsp": DQNTrainer,
        "avg_trainer_class_nfsp": NFSPTrainer,
        "policy_classes_nfsp": {
            "average_policy": NFSPTorchAveragePolicy,
            "delegate_policy": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "anticipatory_param_nfsp": 0.1,
        "get_trainer_config_nfsp": nfsp_40x_dummy_leduc_params_gpu,
        "get_avg_trainer_config_nfsp": nfsp_40x_dummy_leduc_avg_policy_params_gpu,
        "calculate_openspiel_metanash": True,
        "calculate_openspiel_metanash_at_end": False,
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },





    # XFDO universal poker #################

    "12_no_limit_leduc_xfdo_dqn_nfsp_gpu_dynamic_threshold_1_aggressive": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,


        "xfdo_port": 4460 + _GRL_SEED,
        "use_openspiel_restricted_game": False,
        "restricted_game_custom_model": None,
        "xfdo_metanash_method": "nfsp",
        "get_restricted_game_solver": lambda scenario: SolveRestrictedGameDynamicRewardThreshold1(
            scenario=scenario,
            dont_solve_first_n_xfdo_iters=7,
            starting_rew_threshold=1.0,
            min_rew_threshold=0.05,
            min_episodes=200000,
            epsilon=0.05,
            required_fields=None,
        ),

        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "universal_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "universal_poker_stack_size": 12,
        },

        "trainer_class_br": DQNTrainer,
        "policy_classes_br": {
            "metanash": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },

        "get_trainer_config_br": psro_12_no_limit_leduc_params_gpu,
        "get_stopping_condition_br": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),

        "trainer_class_nfsp": DQNTrainer,
        "avg_trainer_class_nfsp": NFSPTrainer,
        "policy_classes_nfsp": {
            "average_policy": NFSPTorchAveragePolicy,
            "delegate_policy": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "anticipatory_param_nfsp": 0.1,
        "get_trainer_config_nfsp": nfsp_12_no_limit_leduc_params_gpu,
        "get_avg_trainer_config_nfsp": nfsp_12_no_limit_leduc_avg_policy_params_gpu,
        "calculate_openspiel_metanash": False,
        "calculate_openspiel_metanash_at_end": False,
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },

    "30_no_limit_leduc_xfdo_dqn_nfsp_gpu_dynamic_threshold_1_aggressive": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,


        "xfdo_port": 4465 + _GRL_SEED,
        "use_openspiel_restricted_game": False,
        "restricted_game_custom_model": None,
        "xfdo_metanash_method": "nfsp",
        "get_restricted_game_solver": lambda scenario: SolveRestrictedGameDynamicRewardThreshold1(
            scenario=scenario,
            dont_solve_first_n_xfdo_iters=7,
            starting_rew_threshold=1.0,
            min_rew_threshold=0.05,
            min_episodes=200000,
            epsilon=0.05,
            required_fields=None,
        ),

        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "universal_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "universal_poker_stack_size": 30,
        },

        "trainer_class_br": DQNTrainer,
        "policy_classes_br": {
            "metanash": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },

        "get_trainer_config_br": psro_30_no_limit_leduc_params_gpu,
        "get_stopping_condition_br": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),

        "trainer_class_nfsp": DQNTrainer,
        "avg_trainer_class_nfsp": NFSPTrainer,
        "policy_classes_nfsp": {
            "average_policy": NFSPTorchAveragePolicy,
            "delegate_policy": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "anticipatory_param_nfsp": 0.1,
        "get_trainer_config_nfsp": nfsp_30_no_limit_leduc_params_gpu,
        "get_avg_trainer_config_nfsp": nfsp_30_no_limit_leduc_avg_policy_params_gpu,
        "calculate_openspiel_metanash": False,
        "calculate_openspiel_metanash_at_end": False,
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },

    "60_no_limit_leduc_xfdo_dqn_nfsp_gpu_dynamic_threshold_1_aggressive": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,


        "xfdo_port": 4470 + _GRL_SEED,
        "use_openspiel_restricted_game": False,
        "restricted_game_custom_model": None,
        "xfdo_metanash_method": "nfsp",
        "get_restricted_game_solver": lambda scenario: SolveRestrictedGameDynamicRewardThreshold1(
            scenario=scenario,
            dont_solve_first_n_xfdo_iters=7,
            starting_rew_threshold=1.0,
            min_rew_threshold=0.05,
            min_episodes=200000,
            epsilon=0.05,
            required_fields=None,
        ),

        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "universal_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "universal_poker_stack_size": 60,
        },

        "trainer_class_br": DQNTrainer,
        "policy_classes_br": {
            "metanash": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },

        "get_trainer_config_br": psro_60_no_limit_leduc_params_gpu,
        "get_stopping_condition_br": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),

        "trainer_class_nfsp": DQNTrainer,
        "avg_trainer_class_nfsp": NFSPTrainer,
        "policy_classes_nfsp": {
            "average_policy": NFSPTorchAveragePolicy,
            "delegate_policy": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "anticipatory_param_nfsp": 0.1,
        "get_trainer_config_nfsp": nfsp_60_no_limit_leduc_params_gpu,
        "get_avg_trainer_config_nfsp": nfsp_60_no_limit_leduc_avg_policy_params_gpu,
        "calculate_openspiel_metanash": False,
        "calculate_openspiel_metanash_at_end": False,
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },

    # VA XFDO universal poker ################################

    "va_12_no_limit_leduc_xfdo_dqn_nfsp_gpu_dynamic_threshold_1_aggressive": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,

        "xfdo_port": 4475 + _GRL_SEED,
        "use_openspiel_restricted_game": True,
        "restricted_game_custom_model": get_valid_action_fcn_class(obs_len=_12_NO_LIMIT_LEDUC_OBS_LEN, action_space_n=13, dummy_actions_multiplier=1),
        "xfdo_metanash_method": "nfsp",
        "get_restricted_game_solver": lambda scenario: SolveRestrictedGameDynamicRewardThreshold1(
            scenario=scenario,
            dont_solve_first_n_xfdo_iters=7,
            starting_rew_threshold=1.0,
            min_rew_threshold=0.05,
            min_episodes=200000,
            epsilon=0.05,
            required_fields=None,
        ),

        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "universal_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "universal_poker_stack_size": 12,
        },

        "trainer_class_br": DQNTrainer,
        "policy_classes_br": {
            "metanash": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },

        "get_trainer_config_br": psro_12_no_limit_leduc_params_gpu,
        "get_stopping_condition_br": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),

        "trainer_class_nfsp": DQNTrainer,
        "avg_trainer_class_nfsp": NFSPTrainer,
        "policy_classes_nfsp": {
            "average_policy": NFSPTorchAveragePolicy,
            "delegate_policy": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "anticipatory_param_nfsp": 0.1,
        "get_trainer_config_nfsp": nfsp_12_no_limit_leduc_params_gpu,
        "get_avg_trainer_config_nfsp": nfsp_12_no_limit_leduc_avg_policy_params_gpu,
        "calculate_openspiel_metanash": False,
        "calculate_openspiel_metanash_at_end": True,  # This is not true for other no limit leduc sizes
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },

    "va_30_no_limit_leduc_xfdo_dqn_nfsp_gpu_dynamic_threshold_1_aggressive": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,

        "xfdo_port": 4480 + _GRL_SEED,
        "use_openspiel_restricted_game": True,
        "restricted_game_custom_model": get_valid_action_fcn_class(obs_len=_30_NO_LIMIT_LEDUC_OBS_LEN, action_space_n=31, dummy_actions_multiplier=1),
        "xfdo_metanash_method": "nfsp",
        "get_restricted_game_solver": lambda scenario: SolveRestrictedGameDynamicRewardThreshold1(
            scenario=scenario,
            dont_solve_first_n_xfdo_iters=7,
            starting_rew_threshold=1.0,
            min_rew_threshold=0.05,
            min_episodes=200000,
            epsilon=0.05,
            required_fields=None,
        ),

        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "universal_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "universal_poker_stack_size": 30,
        },

        "trainer_class_br": DQNTrainer,
        "policy_classes_br": {
            "metanash": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },

        "get_trainer_config_br": psro_30_no_limit_leduc_params_gpu,
        "get_stopping_condition_br": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),

        "trainer_class_nfsp": DQNTrainer,
        "avg_trainer_class_nfsp": NFSPTrainer,
        "policy_classes_nfsp": {
            "average_policy": NFSPTorchAveragePolicy,
            "delegate_policy": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "anticipatory_param_nfsp": 0.1,
        "get_trainer_config_nfsp": nfsp_30_no_limit_leduc_params_gpu,
        "get_avg_trainer_config_nfsp": nfsp_30_no_limit_leduc_avg_policy_params_gpu,
        "calculate_openspiel_metanash": False,
        "calculate_openspiel_metanash_at_end": False,
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },

    "va_60_no_limit_leduc_xfdo_dqn_nfsp_gpu_dynamic_threshold_1_aggressive": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,

        "xfdo_port": 4485 + _GRL_SEED,
        "use_openspiel_restricted_game": True,
        "restricted_game_custom_model": get_valid_action_fcn_class(obs_len=_60_NO_LIMIT_LEDUC_OBS_LEN, action_space_n=61, dummy_actions_multiplier=1),
        "xfdo_metanash_method": "nfsp",
        "get_restricted_game_solver": lambda scenario: SolveRestrictedGameDynamicRewardThreshold1(
            scenario=scenario,
            dont_solve_first_n_xfdo_iters=7,
            starting_rew_threshold=1.0,
            min_rew_threshold=0.05,
            min_episodes=200000,
            epsilon=0.05,
            required_fields=None,
        ),

        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "universal_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "universal_poker_stack_size": 60,
        },

        "trainer_class_br": DQNTrainer,
        "policy_classes_br": {
            "metanash": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },

        "get_trainer_config_br": psro_60_no_limit_leduc_params_gpu,
        "get_stopping_condition_br": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),

        "trainer_class_nfsp": DQNTrainer,
        "avg_trainer_class_nfsp": NFSPTrainer,
        "policy_classes_nfsp": {
            "average_policy": NFSPTorchAveragePolicy,
            "delegate_policy": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "anticipatory_param_nfsp": 0.1,
        "get_trainer_config_nfsp": nfsp_60_no_limit_leduc_params_gpu,
        "get_avg_trainer_config_nfsp": nfsp_60_no_limit_leduc_avg_policy_params_gpu,
        "calculate_openspiel_metanash": False,
        "calculate_openspiel_metanash_at_end": False,
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },

    #### OSHI ZUMO XFDO ######################################################

    "oshi_zumo_xfdo_dqn_nfsp_gpu_dynamic_threshold_1_aggressive": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,

        "xfdo_port": 4490 + _GRL_SEED,
        "use_openspiel_restricted_game": False,
        "restricted_game_custom_model": None,
        "xfdo_metanash_method": "nfsp",
        "get_restricted_game_solver": lambda scenario: SolveRestrictedGameDynamicRewardThreshold1(
            scenario=scenario,
            dont_solve_first_n_xfdo_iters=7,
            starting_rew_threshold=1.0,
            min_rew_threshold=0.05,
            min_episodes=100000,
            epsilon=0.05,
            required_fields=None,
        ),

        "env_class": OshiZumoMultiAgentEnv,
        "env_config": {
            'version': "oshi_zumo",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
        },

        "trainer_class_br": DQNTrainer,
        "policy_classes_br": {
            "metanash": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },

        "get_trainer_config_br": psro_oshi_zumo_dqn_params_like_leduc_gpu,
        "get_stopping_condition_br": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),

        "trainer_class_nfsp": DQNTrainer,
        "avg_trainer_class_nfsp": NFSPTrainer,
        "policy_classes_nfsp": {
            "average_policy": NFSPTorchAveragePolicy,
            "delegate_policy": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "anticipatory_param_nfsp": 0.1,
        "get_trainer_config_nfsp": nfsp_oshi_zumo_avg_policy_params_like_leduc_gpu,
        "get_avg_trainer_config_nfsp": nfsp_oshi_zumo_avg_policy_params_like_leduc_gpu,
        "calculate_openspiel_metanash": False,
        "calculate_openspiel_metanash_at_end": False,
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },


    # 80x dummy leduc ##########################################3

    "80x_dummy_leduc_xfdo_dqn_nfsp_gpu_dynamic_threshold_1_aggressive": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,

        "xfdo_port": 4500 + _GRL_SEED,
        "use_openspiel_restricted_game": False,
        "restricted_game_custom_model": None,
        "xfdo_metanash_method": "nfsp",
        "get_restricted_game_solver": lambda scenario: SolveRestrictedGameDynamicRewardThreshold1(
            scenario=scenario,
            dont_solve_first_n_xfdo_iters=7,
            starting_rew_threshold=1.0,
            min_rew_threshold=0.05,
            min_episodes=200000,
            epsilon=0.05,
            required_fields=["z_avg_policy_exploitability"],
        ),

        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "leduc_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "dummy_action_multiplier": 80,
        },

        "trainer_class_br": DQNTrainer,
        "policy_classes_br": {
            "metanash": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },

        "get_trainer_config_br": psro_80x_dummy_leduc_params_gpu,
        "get_stopping_condition_br": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),

        "trainer_class_nfsp": DQNTrainer,
        "avg_trainer_class_nfsp": NFSPTrainer,
        "policy_classes_nfsp": {
            "average_policy": NFSPTorchAveragePolicy,
            "delegate_policy": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "anticipatory_param_nfsp": 0.1,
        "get_trainer_config_nfsp": nfsp_80x_dummy_leduc_params_gpu,
        "get_avg_trainer_config_nfsp": nfsp_80x_dummy_leduc_avg_policy_params_gpu,
        "calculate_openspiel_metanash": True,
        "calculate_openspiel_metanash_at_end": False,
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },

    "va_80x_dummy_leduc_xfdo_dqn_nfsp_gpu_dynamic_threshold_1_aggressive": {
        "ray_cluster_cpus": _default_if_creating_ray_head(default=8),
        "ray_cluster_gpus": _default_if_creating_ray_head(default=0),
        "ray_object_store_memory_cap_gigabytes": 1,

        "xfdo_port": 4505 + _GRL_SEED,
        "use_openspiel_restricted_game": True,
        "restricted_game_custom_model": get_valid_action_fcn_class(obs_len=_LEDUC_OBS_LEN, action_space_n=3 * 80,
                                                                   dummy_actions_multiplier=1),
        "xfdo_metanash_method": "nfsp",
        "get_restricted_game_solver": lambda scenario: SolveRestrictedGameDynamicRewardThreshold1(
            scenario=scenario,
            dont_solve_first_n_xfdo_iters=7,
            starting_rew_threshold=1.0,
            min_rew_threshold=0.05,
            min_episodes=200000,
            epsilon=0.05,
            required_fields=["z_avg_policy_exploitability"],
        ),

        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "leduc_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "dummy_action_multiplier": 80,
        },

        "trainer_class_br": DQNTrainer,
        "policy_classes_br": {
            "metanash": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicyPatched,
        },

        "get_trainer_config_br": psro_80x_dummy_leduc_params_gpu,
        "get_stopping_condition_br": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(2e4),
            check_plateau_every_n_episodes=int(2e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(1e5),
        ),

        "trainer_class_nfsp": DQNTrainer,
        "avg_trainer_class_nfsp": NFSPTrainer,
        "policy_classes_nfsp": {
            "average_policy": NFSPTorchAveragePolicy,
            "delegate_policy": SimpleQTorchPolicyPatched,
            "best_response": SimpleQTorchPolicyPatched,
        },
        "anticipatory_param_nfsp": 0.1,
        "get_trainer_config_nfsp": nfsp_80x_dummy_leduc_params_gpu,
        "get_avg_trainer_config_nfsp": nfsp_80x_dummy_leduc_avg_policy_params_gpu,
        "calculate_openspiel_metanash": True,
        "calculate_openspiel_metanash_at_end": False,
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },



}