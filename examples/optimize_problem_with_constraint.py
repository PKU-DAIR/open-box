# License: MIT

import numpy as np
import matplotlib.pyplot as plt
from openbox import Optimizer, sp


def mishra(config: sp.Configuration):
    config_dict = config.get_dictionary()
    X = np.array([config_dict['x%d' % i] for i in range(2)])
    x, y = X[0], X[1]
    t1 = np.sin(y) * np.exp((1 - np.cos(x))**2)
    t2 = np.cos(x) * np.exp((1 - np.sin(y))**2)
    t3 = (x - y)**2

    result = dict()
    result['objs'] = [t1 + t2 + t3, ]
    result['constraints'] = [np.sum((X + 5)**2) - 25, ]
    return result


if __name__ == "__main__":
    params = {
        'float': {
            'x0': (-10, 0, -5),
            'x1': (-6.5, 0, -3.25)
        }
    }
    space = sp.Space()
    space.add_variables([
        sp.Real(name, *para) for name, para in params['float'].items()
    ])

    opt = Optimizer(
        mishra,
        space,
        num_constraints=1,
        num_objs=1,
        surrogate_type='gp',
        acq_optimizer_type='random_scipy',
        max_runs=50,
        time_limit_per_trial=10,
        task_id='soc',
    )
    history = opt.run()

    print(history)

    history.plot_convergence(true_minimum=-106.7645367)
    plt.show()
