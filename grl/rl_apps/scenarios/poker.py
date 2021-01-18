
from grl.rl_apps.kuhn_poker_p2sro.config import psro_kuhn_sac_params, psro_kuhn_dqn_params, psro_leduc_dqn_params, psro_fast_leduc_dqn_params
from grl.rl_apps.nfsp.config import nfsp_kuhn_sac_params, nfsp_leduc_dqn_params, nfsp_kuhn_dqn_params

from grl.rl_apps.kuhn_poker_p2sro.poker_multi_agent_env import PokerMultiAgentEnv
from grl.rl_apps.kuhn_poker_p2sro.oshi_zumo_multi_agent_env import OshiZumoMultiAgentEnv
from grl.rl_apps.scenarios.stopping_conditions import SingleBRRewardPlateauStoppingCondition, NoStoppingCondition, TwoPlayerBRRewardsBelowAmtStoppingCondition

from ray.rllib.agents.sac import SACTrainer, SACTorchPolicy
from ray.rllib.agents.dqn import DQNTrainer, DQNTorchPolicy, SimpleQTorchPolicy
from grl.nfsp_rllib.nfsp import NFSPTrainer, NFSPTorchAveragePolicy, get_store_to_avg_policy_buffer_fn


scenarios = {

    # NFSP #########################################################

    "kuhn_nfsp_dqn": {
        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "kuhn_poker",
            "fixed_players": True,
        },
        "trainer_class": DQNTrainer,
        "avg_trainer_class": NFSPTorchAveragePolicy,
        "policy_classes": {
            "average_policy": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicy,
        },
        "get_trainer_config": nfsp_kuhn_dqn_params,
        "anticipatory_param": 0.1,
        "nfsp_get_stopping_condition": lambda: NoStoppingCondition,
        "calculate_openspiel_metanash": True,
        "calc_metanash_every_n_iters": 500,
    },


    "20x_dummy_kuhn_nfsp_dqn": {
        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "kuhn_poker",
            "fixed_players": True,
            "dummy_action_multiplier": 20,
        },
        "trainer_class": DQNTrainer,
        "policy_classes": {
            "average_policy": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicy,
        },
        "get_trainer_config": nfsp_kuhn_dqn_params,
        "anticipatory_param": 0.1,
        "nfsp_get_stopping_condition": lambda: NoStoppingCondition,
        "calculate_openspiel_metanash": True,
        "calc_metanash_every_n_iters": 500,
    },

    "leduc_nfsp_dqn": {
        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "leduc_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
        },
        "trainer_class": DQNTrainer,
        "policy_classes": {
            "average_policy": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicy,
        },
        "get_trainer_config": nfsp_leduc_dqn_params,
        "anticipatory_param": 0.1,
        "nfsp_get_stopping_condition": lambda: NoStoppingCondition,
        "calculate_openspiel_metanash": True,
        "calc_metanash_every_n_iters": 2000,
    },

    "20x_dummy_leduc_nfsp_dqn": {
        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "leduc_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "dummy_action_multiplier": 20,
        },
        "trainer_class": DQNTrainer,
        "policy_classes": {
            "average_policy": NFSPTorchAveragePolicy,
            "best_response": SimpleQTorchPolicy,
        },
        "get_trainer_config": nfsp_leduc_dqn_params,
        "anticipatory_param": 0.1,
        "nfsp_get_stopping_condition": lambda: NoStoppingCondition,
        "calculate_openspiel_metanash": True,
        "calc_metanash_every_n_iters": 2000,
    },

    # CFP #########################################################

    "kuhn_cfp_dqn": {
        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "kuhn_poker",
            "fixed_players": True,
        },
        "trainer_class": DQNTrainer,
        "policy_classes": {
            "best_response": SimpleQTorchPolicy,
        },
        "get_trainer_config": nfsp_kuhn_dqn_params,
        "anticipatory_param": 0.1,
        "nfsp_get_stopping_condition": lambda: NoStoppingCondition,
        "calculate_openspiel_metanash": True,
        "calc_metanash_every_n_iters": 500,
    },

    "20x_dummy_kuhn_cfp_dqn": {
        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "kuhn_poker",
            "fixed_players": True,
            "dummy_action_multiplier": 20,
        },
        "trainer_class": DQNTrainer,
        "policy_classes": {
            "best_response": SimpleQTorchPolicy,
        },
        "get_trainer_config": nfsp_kuhn_dqn_params,
        "anticipatory_param": 0.1,
        "nfsp_get_stopping_condition": lambda: NoStoppingCondition,
        "calculate_openspiel_metanash": True,
        "calc_metanash_every_n_iters": 500,
    },

    "leduc_cfp_dqn": {
        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "leduc_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
        },
        "trainer_class": DQNTrainer,
        "policy_classes": {
            "best_response": SimpleQTorchPolicy,
        },
        "get_trainer_config": nfsp_leduc_dqn_params,
        "anticipatory_param": 0.1,
        "nfsp_get_stopping_condition": lambda: NoStoppingCondition,
        "calculate_openspiel_metanash": True,
        "calc_metanash_every_n_iters": 2000,
    },

    "20x_dummy_leduc_cfp_dqn": {
        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "leduc_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "dummy_action_multiplier": 20,
        },
        "trainer_class": DQNTrainer,
        "policy_classes": {
            "best_response": SimpleQTorchPolicy,
        },
        "get_trainer_config": nfsp_leduc_dqn_params,
        "anticipatory_param": 0.1,
        "nfsp_get_stopping_condition": lambda: NoStoppingCondition,
        "calculate_openspiel_metanash": True,
        "calc_metanash_every_n_iters": 2000,
    },

    # PSRO #########################################################
    
    "kuhn_psro_dqn": {
        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "kuhn_poker",
            "fixed_players": True,
        },
        "trainer_class": DQNTrainer,
        "policy_classes": {
            "metanash": SimpleQTorchPolicy,
            "best_response": SimpleQTorchPolicy,
            "eval": SimpleQTorchPolicy,
        },
        "psro_port": 4100,
        "eval_port": 4200,
        "num_eval_workers": 8,
        "games_per_payoff_eval": 20000,
        "p2sro": False,
        "get_trainer_config": psro_kuhn_dqn_params,
        "psro_get_stopping_condition": lambda: SingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_iters=300,
            check_plateau_every_n_iters=100,
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_iters=20000,
        ),
    },

    "20x_dummy_kuhn_psro_dqn": {
        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "kuhn_poker",
            "fixed_players": True,
            "dummy_action_multiplier": 20,
        },
        "trainer_class": DQNTrainer,
        "policy_classes": {
            "metanash": SimpleQTorchPolicy,
            "best_response": SimpleQTorchPolicy,
            "eval": SimpleQTorchPolicy,
        },
        "psro_port": 4105,
        "eval_port": 4205,
        "num_eval_workers": 8,
        "games_per_payoff_eval": 20000,
        "p2sro": False,
        "get_trainer_config": psro_kuhn_dqn_params,
        "psro_get_stopping_condition": lambda: SingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_iters=300,
            check_plateau_every_n_iters=100,
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_iters=20000,
        ),
    },

    "leduc_psro_dqn": {
        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "leduc_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
        },
        "trainer_class": DQNTrainer,
        "policy_classes": {
            "metanash": SimpleQTorchPolicy,
            "best_response": SimpleQTorchPolicy,
            "eval": SimpleQTorchPolicy,
        },
        "psro_port": 4110,
        "eval_port": 4210,
        "num_eval_workers": 8,
        "games_per_payoff_eval": 20000,
        "p2sro": False,
        "get_trainer_config": psro_leduc_dqn_params,
        "psro_get_stopping_condition": lambda: SingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_iters=3000,
            check_plateau_every_n_iters=1000,
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_iters=20000,
        ),
    },

    "20x_dummy_leduc_psro_dqn": {
        "env_class": PokerMultiAgentEnv,
        "env_config": {
            'version': "leduc_poker",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
            "dummy_action_multiplier": 20,
        },
        "trainer_class": DQNTrainer,
        "policy_classes": {
            "metanash": SimpleQTorchPolicy,
            "best_response": SimpleQTorchPolicy,
            "eval": SimpleQTorchPolicy,
        },
        "psro_port": 4115,
        "eval_port": 4215,
        "num_eval_workers": 8,
        "games_per_payoff_eval": 20000,
        "p2sro": False,
        "get_trainer_config": psro_leduc_dqn_params,
        "psro_get_stopping_condition": lambda: SingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_iters=3000,
            check_plateau_every_n_iters=1000,
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_iters=20000,
        ),
    },

    "oshi_zumo_psro_dqn_leduc_params": {
        "env_class": OshiZumoMultiAgentEnv,
        "env_config": {
            'version': "oshi_zumo",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
        },
        "trainer_class": DQNTrainer,
        "policy_classes": {
            "metanash": SimpleQTorchPolicy,
            "best_response": SimpleQTorchPolicy,
            "eval": SimpleQTorchPolicy,
        },
        "psro_port": 4120,
        "eval_port": 4220,
        "num_eval_workers": 8,
        "games_per_payoff_eval": 20000,
        "p2sro": False,
        "get_trainer_config": psro_leduc_dqn_params,
        "psro_get_stopping_condition": lambda: SingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_iters=3000,
            check_plateau_every_n_iters=1000,
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_iters=20000,
        ),
    },

    "oshi_zumo_psro_dqn_kuhn_params": {
        "env_class": OshiZumoMultiAgentEnv,
        "env_config": {
            'version': "oshi_zumo",
            "fixed_players": True,
            "append_valid_actions_mask_to_obs": True,
        },
        "trainer_class": DQNTrainer,
        "policy_classes": {
            "metanash": SimpleQTorchPolicy,
            "best_response": SimpleQTorchPolicy,
            "eval": SimpleQTorchPolicy,
        },
        "psro_port": 4125,
        "eval_port": 4225,
        "num_eval_workers": 8,
        "games_per_payoff_eval": 20000,
        "p2sro": False,
        "get_trainer_config": psro_kuhn_dqn_params,
        "psro_get_stopping_condition": lambda: SingleBRRewardPlateauStoppingCondition(
            br_policy_id="best_response",
            dont_check_plateau_before_n_iters=300,
            check_plateau_every_n_iters=100,
            minimum_reward_improvement_otherwise_plateaued=0.01,
            max_train_iters=20000,
        ),
    },

}