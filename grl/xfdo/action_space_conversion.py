import json
from typing import Callable, List

from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import EnvObsType, EnvActionType

from grl.utils.strategy_spec import StrategySpec


class RestrictedToBaseGameActionSpaceConverter:

    def __init__(self,
                 delegate_policy: Policy,
                 policy_specs: List[StrategySpec],
                 load_policy_spec_fn: Callable[[Policy, StrategySpec], None]):

        self.delegate_policy = delegate_policy
        self.policy_specs = policy_specs
        self.load_policy_fn = load_policy_spec_fn

    def num_actions(self) -> int:
        return len(self.policy_specs)

    def get_base_game_action(self,
                             obs: EnvObsType,
                             restricted_game_action: int,
                             use_delegate_policy_exploration: bool = False,
                             clip_base_game_actions: bool = False,
                             delegate_policy_state=None) -> EnvActionType:

        if restricted_game_action > self.num_actions():
            raise IndexError(f"restricted game action was {restricted_game_action}"
                             f"while there are only {self.num_actions()} restricted game actions specified.")
        try:
            corresponding_spec_to_action: StrategySpec = self.policy_specs[restricted_game_action]
        except IndexError:
            raise IndexError(
                f"restricted game action of {restricted_game_action} used with space of n={self.num_actions()}")
        self.load_policy_fn(self.delegate_policy, corresponding_spec_to_action)
        base_game_action = self.delegate_policy.compute_single_action(
            obs=obs, state=delegate_policy_state, prev_action=None, prev_reward=None, info=None,
            episode=None, clip_actions=clip_base_game_actions, explore=use_delegate_policy_exploration, timestep=None)
        return base_game_action

    def policy_specs_as_json(self) -> str:
        return json.dumps([spec.to_json() for spec in self.policy_specs])
