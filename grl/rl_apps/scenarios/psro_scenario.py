from typing import Union, Type, Dict, Any, Callable

from ray.rllib import MultiAgentEnv, Policy
from ray.rllib.agents import Trainer
from ray.rllib.utils.typing import ResultDict

from grl.rl_apps.scenarios.scenario import RayScenario
from grl.rl_apps.scenarios.stopping_conditions import StoppingCondition


def psro_default_log_filter(result: ResultDict) -> bool:
    return result["training_iteration"] % 10 == 0


class PSROScenario(RayScenario):

    def __init__(self,
                 name: str,
                 ray_cluster_cpus: Union[int, float],
                 ray_cluster_gpus: Union[int, float],
                 ray_object_store_memory_cap_gigabytes: Union[int, float],
                 env_class: Type[MultiAgentEnv],
                 env_config: Dict[str, Any],
                 mix_metanash_with_uniform_dist_coeff: float,
                 allow_stochastic_best_responses: bool,
                 trainer_class: Type[Trainer],
                 policy_classes: Dict[str, Type[Policy]],
                 num_eval_workers: int,
                 games_per_payoff_eval: int,
                 p2sro: bool,
                 p2sro_payoff_table_exponential_avg_coeff: Union[float, None],
                 p2sro_sync_with_payoff_table_every_n_episodes: Union[int, None],
                 single_agent_symmetric_game: bool,
                 get_trainer_config: Callable[[MultiAgentEnv], Dict[str, Any]],
                 psro_get_stopping_condition: Callable[[], StoppingCondition],
                 calc_exploitability_for_openspiel_env: bool,
                 ray_should_log_result_filter: Callable[[ResultDict], bool] = psro_default_log_filter):
        super().__init__(name=name,
                         ray_cluster_cpus=ray_cluster_cpus,
                         ray_cluster_gpus=ray_cluster_gpus,
                         ray_object_store_memory_cap_gigabytes=ray_object_store_memory_cap_gigabytes,
                         ray_should_log_result_filter=ray_should_log_result_filter)
        self.env_class = env_class
        self.env_config = env_config
        self.mix_metanash_with_uniform_dist_coeff = mix_metanash_with_uniform_dist_coeff
        self.allow_stochastic_best_responses = allow_stochastic_best_responses
        self.trainer_class = trainer_class
        self.policy_classes = policy_classes
        self.num_eval_workers = num_eval_workers
        self.games_per_payoff_eval = games_per_payoff_eval
        self.p2sro = p2sro
        self.p2sro_payoff_table_exponential_avg_coeff = p2sro_payoff_table_exponential_avg_coeff
        self.p2sro_sync_with_payoff_table_every_n_episodes = p2sro_sync_with_payoff_table_every_n_episodes
        self.single_agent_symmetric_game = single_agent_symmetric_game
        self.get_trainer_config = get_trainer_config
        self.psro_get_stopping_condition = psro_get_stopping_condition
        self.calc_exploitability_for_openspiel_env = calc_exploitability_for_openspiel_env
