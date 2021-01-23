from abc import ABC, abstractmethod

import logging

from ray.rllib.utils.typing import PolicyID

logger = logging.getLogger(__name__)


class StoppingCondition(ABC):

    @abstractmethod
    def should_stop_this_iter(self, latest_trainer_result: dict, *args, **kwargs) -> bool:
        pass


class NoStoppingCondition(StoppingCondition):

    def should_stop_this_iter(self, latest_trainer_result: dict, *args, **kwargs) -> bool:
        return False


class SingleBRRewardPlateauStoppingCondition(StoppingCondition):

    def __init__(self,
                 br_policy_id: PolicyID,
                 dont_check_plateau_before_n_iters: int,
                 check_plateau_every_n_iters: int,
                 minimum_reward_improvement_otherwise_plateaued: float,
                 max_train_iters: int = None):

        self.br_policy_id = br_policy_id
        self.dont_check_plateau_before_n_iters = dont_check_plateau_before_n_iters
        self.check_plateau_every_n_iters = check_plateau_every_n_iters
        self.minimum_reward_improvement_otherwise_saturated = minimum_reward_improvement_otherwise_plateaued
        self.max_train_iters = max_train_iters

        self._iters_since_saturation_checks_began = None
        self._last_saturation_check_reward = None

    def _get_reward_to_measure(self, latest_trainer_result):
        return latest_trainer_result["policy_reward_mean"][self.br_policy_id]

    def should_stop_this_iter(self, latest_trainer_result: dict, *args, **kwargs) -> bool:
        should_stop = False

        train_iter = latest_trainer_result["training_iteration"]
        br_reward_this_iter = self._get_reward_to_measure(latest_trainer_result=latest_trainer_result)

        if train_iter >= self.dont_check_plateau_before_n_iters:
            if self._iters_since_saturation_checks_began is None:
                self._iters_since_saturation_checks_began = 0

            if self._iters_since_saturation_checks_began % self.check_plateau_every_n_iters == 0:
                if self._last_saturation_check_reward is not None:
                    improvement_since_last_check = br_reward_this_iter - self._last_saturation_check_reward
                    logger.info(f"Improvement since last saturation check: {improvement_since_last_check}, minimum target is "
                          f"{self.minimum_reward_improvement_otherwise_saturated}.")
                    if improvement_since_last_check < self.minimum_reward_improvement_otherwise_saturated:
                        # We're no longer improving. Assume we have saturated, and stop training.
                        logger.info(f"Improvement target not reached, stopping training if allowed.")
                        should_stop = True
                self._last_saturation_check_reward = br_reward_this_iter
            self._iters_since_saturation_checks_began += 1

        if train_iter >= self.max_train_iters:
            # Regardless of whether we've saturated, we've been training for too long, so we stop.
            logger.info(f"Max training iters reached ({train_iter}). stopping training if allowed.")
            should_stop = True

        return should_stop


class EpisodesSingleBRRewardPlateauStoppingCondition(StoppingCondition):

    def __init__(self,
                 br_policy_id: PolicyID,
                 dont_check_plateau_before_n_episodes: int,
                 check_plateau_every_n_episodes: int,
                 minimum_reward_improvement_otherwise_plateaued: float,
                 max_train_episodes: int = None):

        self.br_policy_id = br_policy_id
        self.dont_check_plateau_before_n_episodes = dont_check_plateau_before_n_episodes
        self.check_plateau_every_n_episodes = check_plateau_every_n_episodes
        self.minimum_reward_improvement_otherwise_saturated = minimum_reward_improvement_otherwise_plateaued
        self.max_train_episodes = max_train_episodes

        self._next_check_after_n_episodes = self.dont_check_plateau_before_n_episodes
        self._last_saturation_check_reward = None

    def _get_reward_to_measure(self, latest_trainer_result):
        return latest_trainer_result["policy_reward_mean"][self.br_policy_id]

    def should_stop_this_iter(self, latest_trainer_result: dict, *args, **kwargs) -> bool:
        should_stop = False

        episodes = latest_trainer_result["episodes_total"]
        br_reward_this_iter = self._get_reward_to_measure(latest_trainer_result=latest_trainer_result)

        if episodes >= self.dont_check_plateau_before_n_episodes:
            if episodes >= self._next_check_after_n_episodes:
                if self._last_saturation_check_reward is not None:
                    improvement_since_last_check = br_reward_this_iter - self._last_saturation_check_reward
                    logger.info(f"Improvement since last saturation check: {improvement_since_last_check}, minimum target is "
                          f"{self.minimum_reward_improvement_otherwise_saturated}.")
                    if improvement_since_last_check < self.minimum_reward_improvement_otherwise_saturated:
                        # We're no longer improving. Assume we have saturated, and stop training.
                        logger.info(f"Improvement target not reached, stopping training if allowed.")
                        should_stop = True
                self._last_saturation_check_reward = br_reward_this_iter
                self._next_check_after_n_episodes = self._next_check_after_n_episodes + self.check_plateau_every_n_episodes

        if episodes >= self.max_train_episodes:
            # Regardless of whether we've saturated, we've been training for too long, so we stop.
            logger.info(f"Max training episodes reached ({episodes}). stopping training if allowed.")
            should_stop = True

        return should_stop



class TimeStepsSingleBRRewardPlateauStoppingCondition(StoppingCondition):

    def __init__(self,
                 br_policy_id: PolicyID,
                 dont_check_plateau_before_n_steps: int,
                 check_plateau_every_n_steps: int,
                 minimum_reward_improvement_otherwise_plateaued: float,
                 max_train_steps: int = None):

        self.br_policy_id = br_policy_id
        self.dont_check_plateau_before_n_steps = dont_check_plateau_before_n_steps
        self.check_plateau_every_n_steps = check_plateau_every_n_steps
        self.minimum_reward_improvement_otherwise_saturated = minimum_reward_improvement_otherwise_plateaued
        self.max_train_steps = max_train_steps

        self._next_check_after_n_steps = self.dont_check_plateau_before_n_steps
        self._last_saturation_check_reward = None

    def _get_reward_to_measure(self, latest_trainer_result):
        return latest_trainer_result["policy_reward_mean"][self.br_policy_id]

    def should_stop_this_iter(self, latest_trainer_result: dict, *args, **kwargs) -> bool:
        should_stop = False

        steps = latest_trainer_result["timesteps_total"]
        br_reward_this_iter = self._get_reward_to_measure(latest_trainer_result=latest_trainer_result)

        if steps >= self.dont_check_plateau_before_n_steps:
            if steps >= self._next_check_after_n_steps:
                if self._last_saturation_check_reward is not None:
                    improvement_since_last_check = br_reward_this_iter - self._last_saturation_check_reward
                    logger.info(f"Improvement since last saturation check: {improvement_since_last_check}, minimum target is "
                          f"{self.minimum_reward_improvement_otherwise_saturated}.")
                    if improvement_since_last_check < self.minimum_reward_improvement_otherwise_saturated:
                        # We're no longer improving. Assume we have saturated, and stop training.
                        logger.info(f"Improvement target not reached, stopping training if allowed.")
                        should_stop = True
                self._last_saturation_check_reward = br_reward_this_iter
                self._next_check_after_n_steps = self._next_check_after_n_steps + self.check_plateau_every_n_steps

        if steps >= self.max_train_steps:
            # Regardless of whether we've saturated, we've been training for too long, so we stop.
            logger.info(f"Max training steps reached ({steps}). stopping training if allowed.")
            should_stop = True

        return should_stop


class TwoPlayerBRRewardsBelowAmtStoppingCondition(StoppingCondition):

    def __init__(self,
                 stop_if_br_avg_rew_falls_below: float,
                 min_episodes: int = None,
                 required_fields_in_last_train_iter = None):
        self.stop_if_br_avg_rew_falls_below = stop_if_br_avg_rew_falls_below
        self.min_episodes = int(min_episodes)
        self.required_fields_in_last_train_iter = required_fields_in_last_train_iter or []

    def should_stop_this_iter(self, latest_trainer_result: dict, *args, **kwargs) -> bool:
        print(f"Stopping Condition Reward threshold is {self.stop_if_br_avg_rew_falls_below}")
        return bool(
                (self.min_episodes is None or latest_trainer_result["episodes_total"] >= self.min_episodes) and
                all(field in latest_trainer_result for field in self.required_fields_in_last_train_iter) and
                latest_trainer_result["avg_br_reward_both_players"] < self.stop_if_br_avg_rew_falls_below
        )






