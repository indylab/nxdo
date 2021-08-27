from typing import Union, Type, Dict, Any, Callable

from ray.rllib import MultiAgentEnv, Policy
from ray.rllib.agents import Trainer
from ray.rllib.utils.typing import ResultDict

from grl.rl_apps.scenarios.scenario import RayScenario
from grl.rl_apps.scenarios.stopping_conditions import StoppingCondition


def nfsp_default_log_filter(result: ResultDict) -> bool:
    return "avg_policy_exploitability" in result or result["training_iteration"] % 100 == 0


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
                 checkpoint_every_n_iters: Union[None, int],
                 ray_should_log_result_filter: Callable[[ResultDict], bool] = nfsp_default_log_filter):
        super().__init__(name=name,
                         ray_cluster_cpus=ray_cluster_cpus,
                         ray_cluster_gpus=ray_cluster_gpus,
                         ray_object_store_memory_cap_gigabytes=ray_object_store_memory_cap_gigabytes,
                         ray_should_log_result_filter=ray_should_log_result_filter)
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
