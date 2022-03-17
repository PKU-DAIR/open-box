# This en code is licensed under the MIT license found in the
# LICENSE file in the root directory of this en tree.

import sys
import numpy as np
from openbox.utils.start_smbo import create_smbo


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


def main():
    bo = create_smbo(branin, **config_dict)


if __name__ == "__main__":
    sys.exit(main())
