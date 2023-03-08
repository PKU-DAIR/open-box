import numpy as np
from openbox.utils.config_space import ConfigurationSpace, UniformFloatHyperparameter
from openbox.optimizer.parallel_smbo import pSMBO

# Define Configuration Space
config_space = ConfigurationSpace()
x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=0)
x2 = UniformFloatHyperparameter("x2", 0, 15, default_value=0)
config_space.add_hyperparameters([x1, x2])

# Define Objective Function
def branin(config):
    x1 = config['x1']
    x2 = config['x2']

    a = 1.
    b = 5.1 / (4. * np.pi ** 2)
    c = 5. / np.pi
    r = 6.
    s = 10.
    t = 1. / (8. * np.pi)
    y = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s

    ret = dict(
        objectives=[y]
    )
    return ret

# Parallel Evaluation on Local Machine
bo = pSMBO(branin,
           config_space,
           parallel_strategy='async',
           batch_size=4,
           batch_strategy='median_imputation',
           num_objectives=1,
           num_constraints=0,
           max_runs=50,
           surrogate_type='gp',
           max_runtime_per_trial=180,
           task_id='parallel')
bo.run()

inc_value = bo.get_incumbents()
print('BO', '=' * 30)
print(inc_value)
print(bo.get_history())
