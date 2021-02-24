from typing import Union, Callable, Type, Dict, Any

from ray.rllib import MultiAgentEnv, Policy
from ray.rllib.agents import Trainer
from ray.rllib.models import ModelV2

from grl.rl_apps.scenarios.scenario import RayScenario, Scenario
from grl.rl_apps.scenarios.stopping_conditions import StoppingCondition
from grl.xfdo.xfdo_manager.manager import SolveRestrictedGame


class NXDOScenario(RayScenario):

    def __init__(self,
                 name: str,
                 ray_cluster_cpus: Union[int, float],
                 ray_cluster_gpus: Union[int, float],
                 ray_object_store_memory_cap_gigabytes: Union[int, float],
                 use_openspiel_restricted_game: bool,
                 get_restricted_game_custom_model: Union[None, Callable[[MultiAgentEnv], Type[ModelV2]]],
                 xdo_metanash_method: str,
                 get_restricted_game_solver: Callable[[Scenario], SolveRestrictedGame],
                 env_class: Type[MultiAgentEnv],
                 env_config: Dict[str, Any],
                 trainer_class_br: Type[Trainer],
                 policy_classes_br: Dict[str, Type[Policy]],
                 get_trainer_config_br: Callable[[MultiAgentEnv], Dict[str, Any]],
                 get_stopping_condition_br: Callable[[], StoppingCondition],
                 trainer_class_nfsp: Type[Trainer],
                 avg_trainer_class_nfsp: Type[Trainer],
                 policy_classes_nfsp: Dict[str, Type[Policy]],
                 anticipatory_param_nfsp: float,
                 get_trainer_config_nfsp: Callable[[MultiAgentEnv], Dict[str, Any]],
                 get_avg_trainer_config_nfsp: Callable[[MultiAgentEnv], Dict[str, Any]],
                 calculate_openspiel_metanash: bool,
                 calculate_openspiel_metanash_at_end: bool,
                 calc_metanash_every_n_iters: Union[None, int],
                 metanash_metrics_smoothing_episodes_override: Union[None, int]):
        super().__init__(name=name,
                         ray_cluster_cpus=ray_cluster_cpus,
                         ray_cluster_gpus=ray_cluster_gpus,
                         ray_object_store_memory_cap_gigabytes=ray_object_store_memory_cap_gigabytes)
        self.use_openspiel_restricted_game = use_openspiel_restricted_game
        self.get_restricted_game_custom_model = get_restricted_game_custom_model
        self.xdo_metanash_method = xdo_metanash_method
        self.get_restricted_game_solver = get_restricted_game_solver
        self.env_class = env_class
        self.env_config = env_config
        self.trainer_class_br = trainer_class_br
        self.policy_classes_br = policy_classes_br
        self.get_trainer_config_br = get_trainer_config_br
        self.get_stopping_condition_br = get_stopping_condition_br
        self.trainer_class_nfsp = trainer_class_nfsp
        self.avg_trainer_class_nfsp = avg_trainer_class_nfsp
        self.policy_classes_nfsp = policy_classes_nfsp
        self.anticipatory_param_nfsp = anticipatory_param_nfsp
        self.get_trainer_config_nfsp = get_trainer_config_nfsp
        self.get_avg_trainer_config_nfsp = get_avg_trainer_config_nfsp
        self.calculate_openspiel_metanash = calculate_openspiel_metanash
        self.calculate_openspiel_metanash_at_end = calculate_openspiel_metanash_at_end
        self.calc_metanash_every_n_iters = calc_metanash_every_n_iters
        self.metanash_metrics_smoothing_episodes_override = metanash_metrics_smoothing_episodes_override