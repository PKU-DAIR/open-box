import os
import sys
import numpy as np

# Import ConfigSpace and different types of parameters

sys.path.append(os.getcwd())
from openbox.facade.bo_facade import BayesianOptimization
from openbox.facade.batch_bo import BatchBayesianOptimization


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
    return ret


# cs = ConfigurationSpace()
# x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=0)
# x2 = UniformFloatHyperparameter("x2", 0, 15, default_value=0)
# cs.add_hyperparameters([x1, x2])

space_dict = {
    "parameters": {
        "x1": {
            "type": "float",
            "bound": [-5, 10],
            "default": 0
        },
        "x2": {
            "type": "float",
            "bound": [0, 15]
        },
    }
}

from openbox.utils.config_space.space_utils import get_config_space_from_dict
cs = get_config_space_from_dict(space_dict)
print(cs)

bo = BayesianOptimization(branin, cs, max_runs=90, time_limit_per_trial=3, logging_dir='logs')
bo.run()
inc_value = bo.get_incumbent()
print('BO', '=' * 30)
print(inc_value)

# Evaluate the random search.
bo = BayesianOptimization(branin, cs, max_runs=90, time_limit_per_trial=3, sample_strategy='random', logging_dir='logs')
bo.run()
inc_value = bo.get_incumbent()
print('RANDOM', '='*30)
print(inc_value)

# Evaluate batch BO.
bo = BatchBayesianOptimization(branin, cs, max_runs=30, batch_size=3,time_limit_per_trial=3, sample_strategy='median_imputation',logging_dir='logs')
bo.run()
inc_value = bo.get_incumbent()
print('MEDIAN IMPUTATION BATCH BO', '='*30)
print(inc_value)

# Evaluate batch BO.
bo = BatchBayesianOptimization(branin, cs, max_runs=30, batch_size=3,time_limit_per_trial=3, sample_strategy='local_penalization',logging_dir='logs')
bo.run()
inc_value = bo.get_incumbent()
print('LOCAL PENALIZATION BATCH BO', '='*30)
print(inc_value)
