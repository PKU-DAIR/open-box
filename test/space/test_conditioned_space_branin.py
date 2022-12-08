import sys
sys.path.insert(0, '.')

from openbox import sp, Optimizer
import numpy as np
import matplotlib.pyplot as plt


def sample_condition(config):
    # We want x1 >= x2
    if config['x1'] < config['x2']:
        return False  # violate condition
    return True  # valid


# Define Search Space
space = sp.ConditionedSpace()
x1 = sp.Real("x1", -5, 10, default_value=0)
x2 = sp.Real("x2", 0, 15, default_value=0)
space.add_variables([x1, x2])
space.set_sample_condition(sample_condition)


# Define Objective Function
def branin(config):
    x1, x2 = config['x1'], config['x2']
    y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2 \
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    return {'objectives': (y,)}


# Run
if __name__ == "__main__":
    opt = Optimizer(
        branin,
        space,
        max_runs=50,
        surrogate_type='gp',
        acq_optimizer_type='random_scipy',
        time_limit_per_trial=30,
        task_id='test_cs',
        random_state=1,
    )
    history = opt.run()

    print(history)

    history.plot_convergence(true_minimum=0.397887)
    plt.show()

