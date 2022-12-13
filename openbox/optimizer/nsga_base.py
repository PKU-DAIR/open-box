# License: MIT

import os
import abc
import time
import numpy as np

from openbox import logger
from openbox.utils.util_funcs import check_random_state
from openbox.utils.constants import MAXINT


class NSGABase(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            objective_function,
            config_space,
            task_id='OpenBox',
            output_dir='logs/',
            random_state=None,
            max_runs=2500,
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
        self.config_space.seed(self.rng.randint(MAXINT))
        self.objective_function = objective_function
        self.max_iterations = max_runs

    def run(self):
        raise NotImplementedError()

    def iterate(self):
        raise NotImplementedError()

    def get_incumbents(self):
        raise NotImplementedError()
