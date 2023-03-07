# License: MIT
import numpy as np
import matplotlib.pyplot as plt
from openbox import create_optimizer
from openbox.utils.constants import SUCCESS


# Define Objective Function
def branin(config):
    x1, x2 = config['x1'], config['x2']
    y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2 \
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    return {'objectives': [y]}


def test_examples_quick_example_with_json():
    config_dict = {
        "optimizer": "SMBO",
        "parameters": {
            "x1": {
                "type": "real",
                "bound": [-5, 10],
                "default": 0
            },
            "x2": {
                "type": "real",
                "bound": [0, 15]
            },
        },
        "advisor_type": "default",
        "max_runs": 50,
        # "surrogate_type": "gp",
        "surrogate_type": "auto",
        "task_id": "quick_example",
        "logging_dir": "logs/pytest/",
    }

    opt = create_optimizer(branin, **config_dict)
    history = opt.run()

    print(history)
    history.plot_convergence(true_minimum=0.397887)
    # plt.show()
    plt.savefig('logs/pytest/quick_example_with_json_convergence.png')
    plt.close()

    assert history.trial_states.count(SUCCESS) == 50
