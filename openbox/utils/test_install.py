# License: MIT

import numpy as np
from openbox import logger
from openbox import Optimizer, space as sp
from openbox.utils.constants import SUCCESS


# Define Objective Function
def branin(config):
    x1, x2 = config['x1'], config['x2']
    y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2 \
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    return {'objectives': [y]}


def run_test():
    logger.info('===== Test Start =====')
    # Define Search Space
    space = sp.Space()
    x1 = sp.Real("x1", -5, 10, default_value=0)
    x2 = sp.Real("x2", 0, 15, default_value=0)
    space.add_variables([x1, x2])

    # Run
    try:
        max_runs = 10
        opt = Optimizer(
            branin,
            space,
            max_runs=max_runs,
            time_limit_per_trial=30,
            task_id='test_install',
        )
        history = opt.run()
    except Exception:
        logger.exception('===== Exception in run_test()! Please check. =====')
    else:
        cnt = history.trial_states.count(SUCCESS)
        if cnt == max_runs:
            logger.info('===== Congratulations! All trials succeeded. =====')
        else:
            logger.info('===== %d/%d trials failed! Please check. =====' % (max_runs-cnt, max_runs))


if __name__ == '__main__':
    run_test()
