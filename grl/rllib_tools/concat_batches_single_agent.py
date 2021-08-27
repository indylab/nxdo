import time
from typing import List
import logging
from ray.rllib import SampleBatch
from ray.rllib.execution.common import _check_sample_batch_type, _get_shared_metrics, SAMPLE_TIMER
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.typing import SampleBatchType

logger = logging.getLogger(__name__)


class ConcatBatchesForSingleAgent:
    """Callable used to merge batches into larger batches for training.

    This should be used with the .combine() operator.

    Examples:
        >>> rollouts = ParallelRollouts(...)
        >>> rollouts = rollouts.combine(ConcatBatches(min_batch_size=10000))
        >>> print(next(rollouts).count)
        10000
    """

    def __init__(self, min_batch_size: int, policy_id_to_count_for: str, drop_samples_for_other_agents=True):
        self.min_batch_size = min_batch_size
        self.policy_id_to_count_for = policy_id_to_count_for
        self.drop_samples_for_other_agents = drop_samples_for_other_agents
        self.buffer = []
        self.count = 0
        self.batch_start_time = None

    def _on_fetch_start(self):
        if self.batch_start_time is None:
            self.batch_start_time = time.perf_counter()

    def __call__(self, batch: MultiAgentBatch) -> List[SampleBatchType]:
        _check_sample_batch_type(batch)
        batch_count = batch.policy_batches[self.policy_id_to_count_for].count
        if self.drop_samples_for_other_agents:
            batch = MultiAgentBatch(policy_batches={self.policy_id_to_count_for: batch.policy_batches[self.policy_id_to_count_for]},
                                    env_steps=batch.policy_batches[self.policy_id_to_count_for].count)

        self.buffer.append(batch)
        self.count += batch_count

        if self.count >= self.min_batch_size:
            if self.count > self.min_batch_size * 2:
                logger.info("Collected more training samples than expected "
                            "(actual={}, expected={}). ".format(
                                self.count, self.min_batch_size) +
                            "This may be because you have many workers or "
                            "long episodes in 'complete_episodes' batch mode.")
            out = SampleBatch.concat_samples(self.buffer)
            timer = _get_shared_metrics().timers[SAMPLE_TIMER]
            timer.push(time.perf_counter() - self.batch_start_time)
            timer.push_units_processed(self.count)
            self.batch_start_time = None
            self.buffer = []
            self.count = 0
            return [out]
        return []