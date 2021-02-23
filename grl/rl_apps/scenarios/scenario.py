from abc import ABC, abstractmethod
from typing import Union, Callable, Dict, Type, Any

from ray.rllib.agents import Trainer
from ray.rllib.env import MultiAgentEnv
from ray.rllib.models import ModelV2
from ray.rllib.policy import Policy

from grl.rl_apps.scenarios.stopping_conditions import StoppingCondition
from grl.rl_apps.xfdo.solve_restricted_game_fns import SolveRestrictedGame


class Scenario(ABC):

    @abstractmethod
    def __init__(self, name: str):
        self.name = name
        self.validate_properties()

    def validate_properties(self):
        # validate name
        for c in self.name:
            if c.isspace() or c in ["\\", "/"] or not c.isprintable():
                raise ValueError("Scenario names must not contain whitespace, '\\', or '/'. "
                                 "It needs to be usable as a directory name. "
                                 f"Yours was '{self.name}'.")


class RayScenario(Scenario, ABC):

    @abstractmethod
    def __init__(self,
                 name: str,
                 ray_cluster_cpus: Union[int, float],
                 ray_cluster_gpus: Union[int, float],
                 ray_object_store_memory_cap_gigabytes: Union[int, float]):
        super().__init__(name=name)
        self.ray_cluster_cpus = ray_cluster_cpus
        self.ray_cluster_gpus = ray_cluster_gpus
        self.ray_object_store_memory_cap_gigabytes = ray_object_store_memory_cap_gigabytes


class NFSPScenario(RayScenario):

    def __init__(self,
                 name: str,
                 ray_cluster_cpus: Union[int, float],
                 ray_cluster_gpus: Union[int, float],
                 ray_object_store_memory_cap_gigabytes: Union[int, float],
                 env_class: Type[MultiAgentEnv],
                 env_config: Dict[str, Any],
                 trainer_class: Type[Trainer],
                 avg_trainer_class: Type[Trainer],
                 policy_classes: Dict[str, Type[Policy]],
                 get_trainer_config: Callable[[MultiAgentEnv], Dict[str, Any]],
                 get_avg_trainer_config: Callable[[MultiAgentEnv], Dict[str, Any]],
                 anticipatory_param: float,
                 nfsp_get_stopping_condition: Callable[[], StoppingCondition],
                 calculate_openspiel_metanash: bool,
                 calculate_openspiel_metanash_at_end: bool,
                 calc_metanash_every_n_iters: int,
                 checkpoint_every_n_iters: Union[None, int]):
        super().__init__(name=name,
                         ray_cluster_cpus=ray_cluster_cpus,
                         ray_cluster_gpus=ray_cluster_gpus,
                         ray_object_store_memory_cap_gigabytes=ray_object_store_memory_cap_gigabytes)
        self.env_class = env_class
        self.env_config = env_config
        self.trainer_class = trainer_class
        self.avg_trainer_class = avg_trainer_class
        self.policy_classes = policy_classes
        self.get_trainer_config = get_trainer_config
        self.get_avg_trainer_config = get_avg_trainer_config
        self.anticipatory_param = anticipatory_param
        self.nfsp_get_stopping_condition = nfsp_get_stopping_condition
        self.calculate_openspiel_metanash = calculate_openspiel_metanash
        self.calculate_openspiel_metanash_at_end = calculate_openspiel_metanash_at_end
        self.calc_metanash_every_n_iters = calc_metanash_every_n_iters
        self.checkpoint_every_n_iters = checkpoint_every_n_iters


class PSROScenario(RayScenario):

    def __init__(self,
                 name: str,
                 ray_cluster_cpus: Union[int, float],
                 ray_cluster_gpus: Union[int, float],
                 ray_object_store_memory_cap_gigabytes: Union[int, float],
                 env_class: Type[MultiAgentEnv],
                 env_config: Dict[str, Any],
                 mix_metanash_with_uniform_dist_coeff: float,
                 trainer_class: Type[Trainer],
                 policy_classes: Dict[str, Type[Policy]],
                 num_eval_workers: int,
                 games_per_payoff_eval: int,
                 p2sro: bool,
                 get_trainer_config: Callable[[MultiAgentEnv], Dict[str, Any]],
                 psro_get_stopping_condition: Callable[[], StoppingCondition]):
        super().__init__(name=name,
                         ray_cluster_cpus=ray_cluster_cpus,
                         ray_cluster_gpus=ray_cluster_gpus,
                         ray_object_store_memory_cap_gigabytes=ray_object_store_memory_cap_gigabytes)
        self.env_class = env_class
        self.env_config = env_config
        self.mix_metanash_with_uniform_dist_coeff = mix_metanash_with_uniform_dist_coeff
        self.trainer_class = trainer_class
        self.policy_classes = policy_classes
        self.num_eval_workers = num_eval_workers
        self.games_per_payoff_eval = games_per_payoff_eval
        self.p2sro = p2sro
        self.get_trainer_config = get_trainer_config
        self.psro_get_stopping_condition = psro_get_stopping_condition


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
