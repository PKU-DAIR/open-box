# License: MIT

import numpy as np
from openbox.core.distributed.worker import Worker
from openbox.utils.config_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter


class BraninWorker(Worker):

    def __init__(self, *args, sleep_interval=0, **kwargs):
        super().__init__(*args, **kwargs)

        self.sleep_interval = sleep_interval

    def compute(self, config, budget, **kwargs):
        import time
        time.sleep(1)
        x1 = config['x1']
        x2 = config['x2']
        a = 1.
        b = 5.1 / (4. * np.pi ** 2)
        c = 5. / np.pi
        r = 6.
        s = 10.
        t = 1. / (8. * np.pi)
        ret = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
        return {'objective_value': float(ret), 'info': budget, 'config': config}

    @staticmethod
    def get_configspace():
        cs = ConfigurationSpace()
        x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=0)
        x2 = UniformFloatHyperparameter("x2", 0, 15, default_value=0)
        cs.add_hyperparameters([x1, x2])
        return cs
