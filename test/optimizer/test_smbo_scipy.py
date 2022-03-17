import os
import sys
import numpy as np

from ConfigSpace.hyperparameters import UniformFloatHyperparameter

sys.path.append(os.getcwd())
from openbox.optimizer.generic_smbo import SMBO
from openbox.utils.config_space import ConfigurationSpace


def branin(x):
    xs = x.get_dictionary()
    x1 = xs['x1']
    x2 = xs['x2']
    a = 1.
    b = 5.1 / (4. * np.pi ** 2)
    c = 5. / np.pi
    r = 6.
    s = 10.
    t = 1. / (8. * np.pi)
    ret = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s

    result = dict()
    result['objs'] = (ret, )

    return result


cs = ConfigurationSpace()
x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=0)
x2 = UniformFloatHyperparameter("x2", 0, 15, default_value=0)
cs.add_hyperparameters([x1, x2])

i = 10
bo = SMBO(branin, cs, advisor_type='default', surrogate_type='gp',
          acq_optimizer_type='local_random', initial_runs=3,
          task_id='local_random_bo', random_state=i, max_runs=31, time_limit_per_trial=3, logging_dir='logs')
bo.run()

bo2 = SMBO(branin, cs, advisor_type='default', surrogate_type='gp',
           acq_optimizer_type='random_scipy', initial_runs=3,
           task_id='random_scipy_bo', random_state=i, max_runs=31, time_limit_per_trial=3, logging_dir='logs')
bo2.run()

print(bo.get_incumbent())
print(bo2.get_incumbent())
