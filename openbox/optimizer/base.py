# License: MIT

import os
import abc
import time
import numpy as np
from typing import List, Optional
from openbox import logger
from openbox.utils.util_funcs import check_random_state, deprecate_kwarg
from openbox.utils.history import History
from manager import TaskManager


class BOBase(object, metaclass=abc.ABCMeta):
    @deprecate_kwarg('time_limit_per_trial', 'max_runtime_per_trial', 'a future version')
    @deprecate_kwarg('runtime_limit', 'max_runtime', 'a future version')
    def __init__(
            self,
            objective_function,
            config_space,
            task_id='OpenBox',
            output_dir='logs/',
            random_state=None,
            initial_runs=3,
            max_runs=100,
            max_runtime=None,
            max_runtime_per_trial=None,
            sample_strategy='bo',
            transfer_learning_history: List[History] = None,
            logger_kwargs: dict = None,
    ):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.task_id = task_id
        _logger_kwargs = {'name': task_id, 'logdir': output_dir}
        _logger_kwargs.update(logger_kwargs or {})
        logger.init(**_logger_kwargs)
        self.rng = check_random_state(random_state)

        self.config_space = config_space
        self.objective_function = objective_function
        self.init_num = initial_runs
        self.max_runs = max_runs
        self.max_runtime = np.inf if max_runtime is None else max_runtime
        self.time_left = self.max_runtime
        self.max_runtime_per_trial = max_runtime_per_trial
        self.iteration_id = 0
        self.sample_strategy = sample_strategy
        self.transfer_learning_history = transfer_learning_history
        self.config_advisor = None
        self.task_manager = getattr(TaskManager, "_instance", None)

    def run(self):
        raise NotImplementedError()

    def iterate(self):
        raise NotImplementedError()

    def get_history(self) -> History:
        assert self.config_advisor is not None
        return self.config_advisor.history

    def get_incumbents(self):
        assert self.config_advisor is not None
        return self.config_advisor.history.get_incumbents()
    
    def _sync_task_manager_history(
        self,
        config,
        objectives: List[float],
        timeout_status: bool,
        traceback_msg: Optional[str],
        elapsed_time: Optional[float],
        extra_info: Optional[dict],
    ) -> None:
        if self.task_manager is None:
            return
        if getattr(self, "num_objectives", 1) != 1:
            return
        if not objectives:
            return
        if not hasattr(self.task_manager, "update_current_task_history"):
            return
        results = {
            'result': {'objective': objectives[0]},
            'timeout': timeout_status,
            'traceback': traceback_msg,
            'elapsed_time': elapsed_time,
            'extra_info': extra_info or {},
        }
        self.task_manager.update_current_task_history(config, results)
