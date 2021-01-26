import os
from grl.rl_apps.kuhn_poker_p2sro.config import psro_kuhn_sac_params, psro_kuhn_dqn_params, psro_leduc_dqn_params, \
    psro_leduc_dqn_params_gpu, psro_kuhn_dqn_params_gpu, psro_oshi_zumo_dqn_params_like_leduc_gpu, psro_20x_dummy_leduc_params_gpu, \
    psro_20x_dummy_leduc_params_gpu_v2


from grl.rl_apps.nfsp.config import nfsp_kuhn_sac_params, nfsp_leduc_dqn_params, nfsp_kuhn_dqn_params, \
    nfsp_oshi_zumo_dqn_params_like_kuhn, nfsp_kuhn_dqn_params_gpu, nfsp_kuhn_avg_policy_params_gpu, nfsp_leduc_avg_policy_params, nfsp_kuhn_avg_policy_params, \
    nfsp_leduc_avg_policy_params_gpu, nfsp_leduc_dqn_params_gpu, nfsp_oshi_zumo_avg_policy_params_like_leduc_gpu, \
    nfsp_oshi_zumo_dqn_params_like_leduc_gpu, nfsp_20x_dummy_leduc_params_gpu, nfsp_20x_dummy_leduc_avg_policy_params_gpu, \
    nfsp_20x_dummy_leduc_avg_policy_params_gpu_v2, nfsp_20x_dummy_leduc_params_gpu_v2

from grl.rl_apps.kuhn_poker_p2sro.poker_multi_agent_env import PokerMultiAgentEnv
from grl.rl_apps.kuhn_poker_p2sro.oshi_zumo_multi_agent_env import OshiZumoMultiAgentEnv, TinyOshiZumoMultiAgentEnv
from grl.rl_apps.scenarios.stopping_conditions import EpisodesSingleBRRewardPlateauStoppingCondition, NoStoppingCondition, StopImmediately, \
    TwoPlayerBRRewardsBelowAmtStoppingCondition, TimeStepsSingleBRRewardPlateauStoppingCondition

from ray.rllib.agents.sac import SACTrainer, SACTorchPolicy
from ray.rllib.agents.dqn import DQNTrainer


from grl.nfsp_rllib.nfsp import NFSPTrainer, NFSPTorchAveragePolicy, get_store_to_avg_policy_buffer_fn
from grl.rllib_tools.modified_policies import SimpleQTorchPolicyPatched, SACTorchPolicyWithBehaviorLogitsOut

from grl.rl_apps.xfdo.solve_restricted_game_fns import SolveRestrictedGameFixedRewardThreshold, SolveRestrictedGameDynamicRewardThreshold1


from ray.rllib.utils import merge_dicts
from ray.rllib.models import MODEL_DEFAULTS
from grl.rllib_tools.valid_actions_fcnet import get_valid_action_fcn_class
from grl.rllib_tools.valid_actions_epsilon_greedy import ValidActionsEpsilonGreedy
from grl.rl_apps.kuhn_poker_p2sro.poker_multi_agent_env import OBS_SHAPES, LEDUC_POKER
from grl.rl_apps.kuhn_poker_p2sro.oshi_zumo_multi_agent_env import OSHI_ZUMO_OBS_LENGTH

_LEDUC_OBS_LEN = OBS_SHAPES[LEDUC_POKER][0]

_GRL_SEED = int(os.getenv("GRL_SEED", 0))


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
    #     "calc_metanash_every_n_iters": 500,
    # },

    "kuhn_nfsp_dqn_gpu": {
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
        "calc_metanash_every_n_iters": 100,
        "checkpoint_every_n_iters": None,
    },

    "leduc_nfsp_dqn_gpu": {
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
        "calc_metanash_every_n_iters": 100,
        "checkpoint_every_n_iters": None,
    },

    "20x_dummy_leduc_nfsp_dqn_gpu": {
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
        "calc_metanash_every_n_iters": 100,
        "checkpoint_every_n_iters": None,
    },


    "20x_dummy_leduc_nfsp_dqn_gpu_v2": {
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
        "calc_metanash_every_n_iters": 100,
        "checkpoint_every_n_iters": None,
    },


    "oshi_zumo_nfsp_dqn_gpu": {
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
        "calc_metanash_every_n_iters": 100,
        "checkpoint_every_n_iters": 200,
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

    "oshi_zumo_psro_dqn_leduc_params_gpu": {
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
        "psro_port": 4115 + _GRL_SEED,
        "eval_port": 4215 + _GRL_SEED,
        "num_eval_workers": 8,
        "games_per_payoff_eval": 1000,
        "p2sro": False,
        "get_trainer_config": psro_oshi_zumo_dqn_params_like_leduc_gpu,
        "psro_get_stopping_condition": lambda: TimeStepsSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_steps=int(1e5),
            check_plateau_every_n_steps=int(1e5),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_steps=int(2e6),
        ),
    },

    "20x_dummy_leduc_psro_dqn_gpu_v2": {
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
        "psro_port": 4120 + _GRL_SEED,
        "eval_port": 4220 + _GRL_SEED,
        "num_eval_workers": 8,
        "games_per_payoff_eval": 3000,
        "p2sro": False,
        "get_trainer_config": psro_20x_dummy_leduc_params_gpu_v2,
        "psro_get_stopping_condition": lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_episodes=int(4e4),
            check_plateau_every_n_episodes=int(4e4),
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_episodes=int(2e5),
        ),
    },


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
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },

    "leduc_xfdo_dqn_nfsp_gpu": {
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
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },

    "20x_dummy_leduc_xfdo_dqn_nfsp_gpu": {
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
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },

    "20x_dummy_leduc_xfdo_dqn_nfsp_gpu_dynamic_threshold_1": {
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
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },


    "20x_dummy_leduc_xfdo_dqn_nfsp_gpu_v2": {
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
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },

    "20x_dummy_leduc_xfdo_dqn_nfsp_gpu_v2_dynamic_threshold_1": {
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
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },

    "leduc_xfdo_dqn_nfsp_gpu_dynamic_threshold_1": {
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
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },


    "va_20x_dummy_leduc_xfdo_dqn_nfsp_gpu_dynamic_threshold_1": {
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
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },

    "20x_dummy_leduc_xfdo_dqn_nfsp_gpu_dynamic_threshold_1_aggressive": {
        "xfdo_port": 4440 + _GRL_SEED,
        "use_openspiel_restricted_game": False,
        "restricted_game_custom_model": None,
        "xfdo_metanash_method": "nfsp",
        "get_restricted_game_solver": lambda scenario: SolveRestrictedGameDynamicRewardThreshold1(
            scenario=scenario,
            dont_solve_first_n_xfdo_iters=7,
            starting_rew_threshold=1.0,
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
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },

    "va_20x_dummy_leduc_xfdo_dqn_nfsp_gpu_dynamic_threshold_1_aggressive": {
        "xfdo_port": 4445 + _GRL_SEED,
        "use_openspiel_restricted_game": True,
        "restricted_game_custom_model": get_valid_action_fcn_class(obs_len=_LEDUC_OBS_LEN, action_space_n=3 * 20, dummy_actions_multiplier=1),
        "xfdo_metanash_method": "nfsp",
        "get_restricted_game_solver": lambda scenario: SolveRestrictedGameDynamicRewardThreshold1(
            scenario=scenario,
            dont_solve_first_n_xfdo_iters=7,
            starting_rew_threshold=1.0,
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
        "calc_metanash_every_n_iters": 50,
        "metanash_metrics_smoothing_episodes_override": 50000,
    },

}