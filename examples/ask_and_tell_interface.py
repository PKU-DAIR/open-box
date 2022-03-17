# License: MIT

import numpy as np
import matplotlib.pyplot as plt
from openbox import Advisor, sp, Observation

# Define Search Space
space = sp.Space()
x1 = sp.Real("x1", -5, 10, default_value=0)
x2 = sp.Real("x2", 0, 15, default_value=0)
space.add_variables([x1, x2])


# Define Objective Function
def branin(config):
    x1, x2 = config['x1'], config['x2']
    y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2 \
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    return {'objs': (y,)}


# Run
if __name__ == "__main__":
    advisor = Advisor(
        space,
        # surrogate_type='gp',
        surrogate_type='auto',
        task_id='quick_start',
    )

    MAX_RUNS = 50
    for i in range(MAX_RUNS):
        # ask
        config = advisor.get_suggestion()
        # evaluate
        ret = branin(config)
        # tell
        observation = Observation(config=config, objs=ret['objs'])
        advisor.update_observation(observation)
        print('===== ITER %d/%d: %s.' % (i+1, MAX_RUNS, observation))

    history = advisor.get_history()
    print(history)

    history.plot_convergence(true_minimum=0.397887)
    plt.show()

    # install pyrfr to use get_importance()
    # print(history.get_importance())

    # history.visualize_jupyter()
