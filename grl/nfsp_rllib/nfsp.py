
import logging
from typing import Optional, Type, Callable
import ray
from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.replay_ops import StoreToReplayBuffer
from ray.rllib.execution.train_ops import TrainOneStep
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import TrainerConfigDict
from ray.rllib.utils.typing import SampleBatchType
from ray.util.iter_metrics import SharedMetrics
from ray.util.iter import LocalIterator, _NextValueNotReady

from grl.nfsp_rllib.reservoir_replay_buffer import ReservoirReplayActor
from grl.nfsp_rllib.nfsp_torch_avg_policy import NFSPTorchAveragePolicy
logger = logging.getLogger(__name__)

# yapf: disable
# __sphinx_doc_begin__
DEFAULT_CONFIG = with_common_config({
    "replay_sequence_length": 1,

    # === Exploration Settings ===
    "exploration_config": {
        # The Exploration class to use.
        "type": "StochasticSampling",
    },
    "evaluation_config": {
        "explore": False,
    },
    "explore": False,

    # === Replay buffer ===
    # Size of the replay buffer. Note that if async_updates is set, then
    # each worker will have a replay buffer of this size.
    "buffer_size": int(2e6),

    # Whether to LZ4 compress observations
    "compress_observations": False,
    # Callback to run before learning on a multi-agent batch of experiences.
    "before_learn_on_batch": None,

    # === Optimization ===
    # Learning rate for adam optimizer
    "lr": 0.01,
    # Learning rate schedule
    "lr_schedule": None,
    # Adam epsilon hyper parameter
    "adam_epsilon": 1e-8,
    # If not None, clip gradients during optimization at this value
    "grad_clip": None,
    # How many steps of the model to sample before learning starts.
    "learning_starts": 2000,

    # Size of a batch sampled from replay buffer for training. Note that
    # if async_updates is set, then each worker returns gradients for a
    # batch of this size.
    "train_batch_size": 128,

    # Minimum env steps to optimize for per train call. This value does
    # not affect learning, only the length of train iterations.
    "timesteps_per_iteration": 0,

    # === Parallelism ===
    # Number of workers for collecting samples with. This only makes sense
    # to increase if your environment is particularly slow to sample, or if
    # you"re using the Async or Ape-X optimizers.
    "num_workers": 0,
    # Prevent iterations from going lower than this time span
    "min_iter_time_s": 0,
})
# __sphinx_doc_end__
# yapf: enable


def validate_config(config: TrainerConfigDict) -> None:
    """Checks and updates the config based on settings.

    Rewrites rollout_fragment_length to take into account n_step truncation.
    """

    pass


def execution_plan(workers: WorkerSet,
                   config: TrainerConfigDict) -> LocalIterator[dict]:
    """Execution plan of the DQN algorithm. Defines the distributed dataflow.

    Args:
        workers (WorkerSet): The WorkerSet for training the Polic(y/ies)
            of the Trainer.
        config (TrainerConfigDict): The trainer's configuration dict.

    Returns:
        LocalIterator[dict]: A local iterator over training metrics.
    """

    replay_buffer_actor = ReservoirReplayActor.remote(
        num_shards=1,
        learning_starts=config["learning_starts"],
        buffer_size=config["buffer_size"],
        replay_batch_size=config["train_batch_size"],
        replay_mode=config["multiagent"]["replay_mode"],
        replay_sequence_length=config["replay_sequence_length"],
        )

    # Store a handle for the replay buffer actor in the local worker
    workers.local_worker().replay_buffer_actor = replay_buffer_actor

    # Read and train on experiences from the replay buffer. Every batch
    # returned from the Replay iterator is passed to TrainOneStep to
    # take a SGD step.
    post_fn = config.get("before_learn_on_batch") or (lambda b, *a: b)

    print("running replay op..")

    def gen_replay(_):
        while True:
            item = ray.get(replay_buffer_actor.replay.remote())
            if item is None:
                yield _NextValueNotReady()
            else:
                yield item

    replay_op = LocalIterator(gen_replay, SharedMetrics())\
        .for_each(lambda x: post_fn(x, workers, config))\
        .for_each(TrainOneStep(workers))

    replay_op = StandardMetricsReporting(replay_op, workers, config)

    replay_op = map(lambda x: x if not isinstance(x, _NextValueNotReady) else {}, replay_op)

    return replay_op

def get_policy_class(config: TrainerConfigDict) -> Optional[Type[Policy]]:
    """Policy class picker function. Class is chosen based on DL-framework.

    Args:
        config (TrainerConfigDict): The trainer's configuration dict.

    Returns:
        Optional[Type[Policy]]: The Policy class to use with DQNTrainer.
            If None, use `default_policy` provided in build_trainer().
    """
    if config["framework"] == "torch":
        return NFSPTorchAveragePolicy
    else:
        raise NotImplementedError(f"NFSP average policy for framework: {config['framework']} not implemented.")

NFSPTrainer = build_trainer(
    name="NFSPTrainer",
    default_policy=NFSPTorchAveragePolicy,
    get_policy_class=get_policy_class,
    default_config=DEFAULT_CONFIG,
    validate_config=validate_config,
    execution_plan=execution_plan,
)


def get_store_to_avg_policy_buffer_fn(nfsp_trainer: NFSPTrainer) -> Callable[[SampleBatchType], SampleBatchType]:
    replay_buffer_actor = nfsp_trainer.workers.local_worker().replay_buffer_actor
    return StoreToReplayBuffer(actors=[replay_buffer_actor])

