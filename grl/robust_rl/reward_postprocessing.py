from abc import ABC, abstractmethod
from typing import List, Dict

import numpy as np
from ray.rllib.utils.typing import AgentID


class TrajectoryRewardPostProcessing(ABC):

    @abstractmethod
    def __call__(self,
                 agent_id_for_trajectory: AgentID,
                 trajectory_rewards: List[float],
                 agent_total_rewards: Dict[AgentID, float]) -> List[float]:
        raise NotImplementedError


class NoOpPostProcessing(TrajectoryRewardPostProcessing):

    def __call__(self,
                 agent_id_for_trajectory: AgentID,
                 trajectory_rewards: List[float],
                 agent_total_rewards: Dict[AgentID, float]) -> List[float]:
        return trajectory_rewards


class StandardRobustRLPostProcessing(TrajectoryRewardPostProcessing):

    def __init__(self,
                 adversary_agent_id: AgentID,
                 protagonist_agent_id: AgentID):

        self.adversary_agent_id = adversary_agent_id
        self.protagonist_agent_id = protagonist_agent_id

    def __call__(self,
                 agent_id_for_trajectory: AgentID,
                 trajectory_rewards: List[float],
                 agent_total_rewards: Dict[AgentID, float]) -> List[float]:
        if agent_id_for_trajectory == self.protagonist_agent_id:
            # no post processing necessary for protagonist
            return trajectory_rewards
        elif agent_id_for_trajectory == self.adversary_agent_id:
            # reward for adversary is zero until the end of the trajectory,
            # at which it's the negative of the protagonist's total reward.
            new_trajectory_rewards = [0.0] * len(trajectory_rewards)
            new_trajectory_rewards[-1] = -1 * agent_total_rewards[self.protagonist_agent_id]
            return new_trajectory_rewards
        else:
            raise NotImplementedError(f"Unknown agent id for reward postprocessing: {agent_id_for_trajectory}")


class ConstrainedRobustRLSingleRunPostProcessingAdversaryOnly(TrajectoryRewardPostProcessing):

    def __init__(self,
                 adversary_agent_id: AgentID,
                 protagonist_agent_id: AgentID,
                 antagonist_agent_id: AgentID,
                 feasibility_reward_threshold: float,
                 infeasibility_penalty_magnitude: float):

        self.adversary_agent_id = adversary_agent_id
        self.protagonist_agent_id = protagonist_agent_id
        self.antagonist_agent_id = antagonist_agent_id
        self.feasibility_reward_threshold = feasibility_reward_threshold

        if infeasibility_penalty_magnitude < 0.0:
            raise ValueError(
                "The infeasibility_penalty_magnitude should be a positive number denoting "
                "how much the adversary/protagonist team is penalized by for providing infeasible conditions.")
        self.infeasibility_penalty_magnitude = infeasibility_penalty_magnitude

    def __call__(self,
                 agent_id_for_trajectory: AgentID,
                 trajectory_rewards: List[float],
                 agent_total_rewards: Dict[AgentID, float]) -> List[float]:

        if agent_id_for_trajectory == self.protagonist_agent_id:
            return trajectory_rewards
        elif agent_id_for_trajectory == self.antagonist_agent_id:
            return trajectory_rewards
        elif agent_id_for_trajectory == self.adversary_agent_id:
            protagonist_single_episode_reward: float = agent_total_rewards[self.protagonist_agent_id]
            antagonist_single_episode_reward: float = agent_total_rewards[self.antagonist_agent_id]

            antagonist_clipped_reward = min(0.0, antagonist_single_episode_reward - self.feasibility_reward_threshold)
            antagonist_penalized_reward = antagonist_clipped_reward * self.infeasibility_penalty_magnitude

            adversary_new_trajectory_rewards = np.zeros_like(trajectory_rewards)
            adversary_new_trajectory_rewards[-1] = antagonist_penalized_reward - protagonist_single_episode_reward
            return list(adversary_new_trajectory_rewards)
        else:
            raise NotImplementedError(f"Unknown agent id for reward postprocessing: {agent_id_for_trajectory}")


class ConstrainedRobustRLSingleRunPostProcessingRev1(TrajectoryRewardPostProcessing):

    def __init__(self,
                 adversary_agent_id: AgentID,
                 protagonist_agent_id: AgentID,
                 antagonist_agent_id: AgentID,
                 feasibility_reward_threshold: float,
                 infeasibility_penalty_magnitude: float):

        self.adversary_agent_id = adversary_agent_id
        self.protagonist_agent_id = protagonist_agent_id
        self.antagonist_agent_id = antagonist_agent_id
        self.feasibility_reward_threshold = feasibility_reward_threshold

        if infeasibility_penalty_magnitude < 0.0:
            raise ValueError(
                "The infeasibility_penalty_magnitude should be a positive number denoting "
                "how much the adversary/protagonist team is penalized by for providing infeasible conditions.")
        self.infeasibility_penalty_magnitude = infeasibility_penalty_magnitude

    def __call__(self,
                 agent_id_for_trajectory: AgentID,
                 trajectory_rewards: List[float],
                 agent_total_rewards: Dict[AgentID, float]) -> List[float]:

        protagonist_single_episode_reward: float = agent_total_rewards[self.protagonist_agent_id]
        antagonist_single_episode_reward: float = agent_total_rewards[self.antagonist_agent_id]

        antagonist_clipped_reward = min(0.0, antagonist_single_episode_reward - self.feasibility_reward_threshold)
        antagonist_penalized_reward = antagonist_clipped_reward * self.infeasibility_penalty_magnitude

        if agent_id_for_trajectory == self.protagonist_agent_id:
            protag_new_trajectory_rewards = np.zeros_like(trajectory_rewards)
            protag_new_trajectory_rewards[-1] = protagonist_single_episode_reward - antagonist_penalized_reward
            return list(protag_new_trajectory_rewards)
        elif agent_id_for_trajectory == self.antagonist_agent_id:
            antag_new_trajectory_rewards = np.zeros_like(trajectory_rewards)
            antag_new_trajectory_rewards[-1] = antagonist_penalized_reward - protagonist_single_episode_reward
            return list(antag_new_trajectory_rewards)
        elif agent_id_for_trajectory == self.adversary_agent_id:
            adversary_new_trajectory_rewards = np.zeros_like(trajectory_rewards)
            adversary_new_trajectory_rewards[-1] = antagonist_penalized_reward - protagonist_single_episode_reward
            return list(adversary_new_trajectory_rewards)
        else:
            raise NotImplementedError(f"Unknown agent id for reward postprocessing: {agent_id_for_trajectory}")


class ConstrainedRobustRLSingleRunPostProcessingRev2(TrajectoryRewardPostProcessing):

    def __init__(self,
                 adversary_agent_id: AgentID,
                 protagonist_agent_id: AgentID,
                 antagonist_agent_id: AgentID,
                 feasibility_reward_threshold: float,
                 infeasibility_penalty_magnitude: float):

        self.adversary_agent_id = adversary_agent_id
        self.protagonist_agent_id = protagonist_agent_id
        self.antagonist_agent_id = antagonist_agent_id
        self.feasibility_reward_threshold = feasibility_reward_threshold

        if infeasibility_penalty_magnitude < 0.0:
            raise ValueError(
                "The infeasibility_penalty_magnitude should be a positive number denoting "
                "how much the adversary/protagonist team is penalized by for providing infeasible conditions.")
        self.infeasibility_penalty_magnitude = infeasibility_penalty_magnitude

    def __call__(self,
                 agent_id_for_trajectory: AgentID,
                 trajectory_rewards: List[float],
                 agent_total_rewards: Dict[AgentID, float]) -> List[float]:

        protagonist_single_episode_reward: float = agent_total_rewards[self.protagonist_agent_id]
        antagonist_single_episode_reward: float = agent_total_rewards[self.antagonist_agent_id]

        antagonist_clipped_reward = min(0.0, antagonist_single_episode_reward - self.feasibility_reward_threshold)
        antagonist_penalized_reward = antagonist_clipped_reward * self.infeasibility_penalty_magnitude

        # difference between what the antag got after reward processing and what it originally got in the base env.
        antag_processed_reward_diff = antagonist_penalized_reward - antagonist_single_episode_reward

        if agent_id_for_trajectory == self.protagonist_agent_id:
            protag_traj_len = len(trajectory_rewards)
            protag_new_trajectory_rewards = np.asarray(trajectory_rewards, dtype=np.float32)

            # subtract antagonist reward all at once at the end (or beginning?),
            # or evenly over the whole trajectory for protagonist?
            protag_new_trajectory_rewards -= (antagonist_penalized_reward / protag_traj_len)  # apply evenly
            # protag_new_trajectory_rewards[0] -= antagonist_penalized_reward  # apply at beginning
            # protag_new_trajectory_rewards[-1] -= antagonist_penalized_reward  # apply at end

            assert sum(protag_new_trajectory_rewards) == protagonist_single_episode_reward - antagonist_penalized_reward
            return list(protag_new_trajectory_rewards)

        elif agent_id_for_trajectory == self.antagonist_agent_id:
            antag_traj_len = len(trajectory_rewards)
            antag_new_trajectory_rewards = np.asarray(trajectory_rewards).copy()

            # apply all feasibility penalties for antagonist at the end of trajectory?
            antag_new_trajectory_rewards[-1] += antag_processed_reward_diff

            # subtract protagonist reward all at once at the end (or beginning?),
            # or evenly over the whole trajectory for antagonist?
            antag_new_trajectory_rewards -= (protagonist_single_episode_reward / antag_traj_len)  # apply evenly
            # antag_new_trajectory_rewards[0] -= protagonist_single_episode_reward # apply at beginning
            # antag_new_trajectory_rewards[-1] -= protagonist_single_episode_reward # apply at end

            assert sum(antag_new_trajectory_rewards) == antagonist_penalized_reward - protagonist_single_episode_reward
            return antag_new_trajectory_rewards

        elif agent_id_for_trajectory == self.adversary_agent_id:
            adversary_new_trajectory_rewards = np.zeros_like(trajectory_rewards)
            adversary_new_trajectory_rewards[-1] = antagonist_penalized_reward - protagonist_single_episode_reward
            return list(adversary_new_trajectory_rewards)
        else:
            raise NotImplementedError(f"Unknown agent id for reward postprocessing: {agent_id_for_trajectory}")


class PAIREDSingleRunPostProcessing(TrajectoryRewardPostProcessing):

    def __init__(self,
                 adversary_agent_id: AgentID,
                 protagonist_agent_id: AgentID,
                 antagonist_agent_id: AgentID):

        self.adversary_agent_id = adversary_agent_id
        self.protagonist_agent_id = protagonist_agent_id
        self.antagonist_agent_id = antagonist_agent_id

    def __call__(self,
                 agent_id_for_trajectory: AgentID,
                 trajectory_rewards: List[float],
                 agent_total_rewards: Dict[AgentID, float]) -> List[float]:

        protagonist_single_episode_reward: float = agent_total_rewards[self.protagonist_agent_id]
        antagonist_single_episode_reward: float = agent_total_rewards[self.antagonist_agent_id]

        if agent_id_for_trajectory == self.protagonist_agent_id:
            protag_new_trajectory_rewards = np.zeros_like(trajectory_rewards)
            protag_new_trajectory_rewards[-1] = protagonist_single_episode_reward - antagonist_single_episode_reward
            return list(protag_new_trajectory_rewards)
        elif agent_id_for_trajectory == self.antagonist_agent_id:
            antag_new_trajectory_rewards = np.zeros_like(trajectory_rewards)
            antag_new_trajectory_rewards[-1] = antagonist_single_episode_reward - protagonist_single_episode_reward
            return list(antag_new_trajectory_rewards)
        elif agent_id_for_trajectory == self.adversary_agent_id:
            adversary_new_trajectory_rewards = np.zeros_like(trajectory_rewards)
            adversary_new_trajectory_rewards[-1] = antagonist_single_episode_reward - protagonist_single_episode_reward
            return list(adversary_new_trajectory_rewards)
        else:
            raise NotImplementedError(f"Unknown agent id for reward postprocessing: {agent_id_for_trajectory}")


class PAIREDSingleRunPostProcessingAdversaryOnly(TrajectoryRewardPostProcessing):

    def __init__(self,
                 adversary_agent_id: AgentID,
                 protagonist_agent_id: AgentID,
                 antagonist_agent_id: AgentID):

        self.adversary_agent_id = adversary_agent_id
        self.protagonist_agent_id = protagonist_agent_id
        self.antagonist_agent_id = antagonist_agent_id

    def __call__(self,
                 agent_id_for_trajectory: AgentID,
                 trajectory_rewards: List[float],
                 agent_total_rewards: Dict[AgentID, float]) -> List[float]:

        if agent_id_for_trajectory == self.protagonist_agent_id:
            return trajectory_rewards
        elif agent_id_for_trajectory == self.antagonist_agent_id:
            return trajectory_rewards
        elif agent_id_for_trajectory == self.adversary_agent_id:
            protagonist_single_episode_reward: float = agent_total_rewards[self.protagonist_agent_id]
            antagonist_single_episode_reward: float = agent_total_rewards[self.antagonist_agent_id]

            adversary_new_trajectory_rewards = np.zeros_like(trajectory_rewards)
            adversary_new_trajectory_rewards[-1] = antagonist_single_episode_reward - protagonist_single_episode_reward
            return list(adversary_new_trajectory_rewards)
        else:
            raise NotImplementedError(f"Unknown agent id for reward postprocessing: {agent_id_for_trajectory}")
