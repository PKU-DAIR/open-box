# License: MIT

import os
import abc
import time
import numpy as np
from openbox.utils.util_funcs import check_random_state
from openbox.utils.logging_utils import setup_logger, get_logger
from openbox.utils.constants import MAXINT


class NSGABase(object, metaclass=abc.ABCMeta):
    def __init__(self, objective_function, config_space, task_id='task_id', output_dir='logs/',
                 random_state=None, max_runs=2500):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.task_id = task_id
        _time_stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        _logger_id = '%s' % task_id
        self.logger_name = None
        self.logger = self._get_logger(_logger_id)
        self.rng = check_random_state(random_state)

        self.config_space = config_space
        self.config_space.seed(self.rng.randint(MAXINT))
        self.objective_function = objective_function
        self.max_iterations = max_runs

    def run(self):
        raise NotImplementedError()

    def iterate(self):
        raise NotImplementedError()

    def get_incumbent(self):
        raise NotImplementedError()

    def _get_logger(self, name):
        logger_name = 'OpenBox-%s' % name
        self.logger_name = os.path.join(self.output_dir, '%s.log' % str(logger_name))
        setup_logger(self.logger_name)
        return get_logger(logger_name)
