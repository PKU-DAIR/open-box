import os
import sys
import numpy as np

sys.path.append(os.getcwd())
from openbox.optimizer.generic_smbo import SMBO
from openbox.utils.config_space import ConfigurationSpace, UniformFloatHyperparameter


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
    return {'objs': (ret,)}


cs = ConfigurationSpace()
x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=0)
x2 = UniformFloatHyperparameter("x2", 0, 15, default_value=0)
cs.add_hyperparameters([x1, x2])

seed = np.random.randint(100)

# random search
# bo = SMBO(branin, cs, sample_strategy='random', max_runs=100, task_id='mcei', random_state=seed)

bo = SMBO(branin, cs,
          advisor_type='mcadvisor',
          acq_type='mcei',
          mc_times=10,
          max_runs=50,
          task_id='mcei', random_state=seed)
bo.run()
inc_value = bo.get_incumbent()
print('BO', '=' * 30)
print(inc_value)
