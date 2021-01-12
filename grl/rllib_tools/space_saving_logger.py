import logging
from typing import Dict, Callable
from ray.tune.result import NODE_IP

from ray.tune.logger import UnifiedLogger

logger = logging.getLogger(__name__)


class SpaceSavingLogger(UnifiedLogger):
    """Unified result logger for TensorBoard, rllab/viskit, plain json.

    Arguments:
        config: Configuration passed to all logger creators.
        logdir: Directory for all logger creators to log to.
        loggers (list): List of logger creators. Defaults to CSV, Tensorboard,
            and JSON loggers.
        sync_function (func|str): Optional function for syncer to run.
            See ray/python/ray/tune/syncer.py
        should_log_result_fn: (func) Callable that takes in a train result and outputs
            whether the result should be logged (bool). Used to save space by only logging important results.
    """

    def __init__(self,
                 config,
                 logdir,
                 trial=None,
                 loggers=None,
                 sync_function=None,
                 should_log_result_fn: Callable[[Dict], bool] = None):

        super(SpaceSavingLogger, self).__init__(config=config,
                                                logdir=logdir,
                                                trial=trial,
                                                loggers=loggers,
                                                sync_function=sync_function)

        self.should_log_result_fn = should_log_result_fn

    def on_result(self, result):
        should_log_result = True
        if self.should_log_result_fn is not None:
            should_log_result = self.should_log_result_fn(result)

        if should_log_result:
            for _logger in self._loggers:
                _logger.on_result(result)
            self._log_syncer.set_worker_ip(result.get(NODE_IP))
            self._log_syncer.sync_down_if_needed()
