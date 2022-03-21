# License: MIT

import numpy as np
import matplotlib.pyplot as plt
from openbox import Advisor, sp, Observation, EA_Advisor
from openbox.core.ea.differential_ea_advisor import DifferentialEAAdvisor
from openbox.core.ea.regularized_ea_advisor import RegularizedEAAdvisor
from openbox.core.ea.adaptive_ea_advisor import AdaptiveEAAdvisor


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
    advisor = AdaptiveEAAdvisor(
        config_space = space,
        task_id='default_task_id',
    )

    used = 1
    MAX_RUNS = 50
    if used == 0:
        for i in range(MAX_RUNS):
            # ask
            config = advisor.get_suggestion()
            # evaluate
            ret = branin(config)
            # tell
            observation = Observation(config=config, objs=ret['objs'])
            advisor.update_observation(observation)
            print('===== ITER %d/%d: %s.' % (i+1, MAX_RUNS, observation))
    else:
        for i in range(MAX_RUNS):
            # ask
            configs = advisor.get_suggestions()
            observations = []
            # evaluate
            for config in configs:
                ret = branin(config)
                observations.append(Observation(config=config, objs=ret['objs']))
            # tell
            advisor.update_observations(observations)
            print('===== ITER %d/%d, %d configs.' % (i+1, MAX_RUNS, len(configs)))

    history = advisor.get_history()
    print(history)

    history.plot_convergence(true_minimum=0.397887)
    plt.show()

    # install pyrfr to use get_importance()
    # print(history.get_importance())

    # history.visualize_jupyter()
