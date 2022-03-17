import numpy as np
from openbox.utils.start_smbo import create_smbo
import matplotlib.pyplot as plt


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


config_dict = {
    "optimizer": "SMBO",
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
    },
    "advisor_type": 'default',
    "max_runs": 50,
    "surrogate_type": 'gp',
    "time_limit_per_trial": 5,
    "logging_dir": 'logs',
    "task_id": 'hp1'
}

bo = create_smbo(branin, **config_dict)
history = bo.run()
inc_value = bo.get_incumbent()
print('BO', '=' * 30)
print(inc_value)

print(history)
history.plot_convergence(true_minimum=0.397887)
plt.show()
# plt.savefig('logs/plot_convergence_branin.png')

# history.visualize_jupyter()
