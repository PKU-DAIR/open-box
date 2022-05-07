# License: MIT

import numpy as np
import matplotlib.pyplot as plt
from openbox import Advisor, sp, Observation, EA_Advisor
from openbox.core.ea.nsga2_ea_advisor import NSGA2EAdvisor
from openbox.core.ea.differential_ea_advisor import DifferentialEAAdvisor
from openbox.core.ea.regularized_ea_advisor import RegularizedEAAdvisor
from openbox.core.ea.adaptive_ea_advisor import AdaptiveEAAdvisor
from openbox.benchmark.objective_functions.synthetic import BraninCurrin

# Define Search Space
# space = sp.Space()
# x1 = sp.Real("x1", -5, 10, default_value=0)
# x2 = sp.Real("x2", 0, 15, default_value=0)
# space.add_variables([x1, x2])

space_num = 5
space_range = 5.12

space = sp.Space()
x = []
for i in range(1, space_num + 1):
    x.append(sp.Real('x' + str(i), -space_range, space_range, default_value=0))
space.add_variables(x)

# Define Objective Function
def sumsqr(config):
    x = [config['x' + str(t)] for t in range(1, space_num + 1)]
    y = sum([(i+1) * (x[i] ** 2) for i in range(space_num)])
    return {'objs': (y,)}

def branin(config):
    x1, x2 = config['x1'], config['x2']
    y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2 \
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    return {'objs': (y,)}

def schwefel(config):
    x = [config['x' + str(t)] for t in range(1, space_num + 1)]
    y = 418.9829 * space_num
    for xi in x:
        y -= xi * np.sin(np.sqrt(np.abs(xi)))
    return {'objs': (y,)}

def rastrigin(config):
    x = [config['x' + str(t)] for t in range(1, space_num + 1)]
    y = space_num * 10
    for xi in x:
        y += (xi ** 2) - 10 * np.cos(2 * np.pi * xi)
    return {'objs': (y,)}

def branincurrin(config):
    x1, x2 = config['x1'], config['x2']
    y1 = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2 \
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    y2 = (1 - np.exp(-1 / (2 * x2))) * (2300 * x1 ** 3 + 1900 * x1 ** 2 + 2092 * x1 + 60) \
         / (100 * x1 ** 3 + 500 * x1 ** 2 + 4 * x1 + 20)
    ret = {}
    ret['objs'] = [y1, y2]
    return ret

# Run

objfunc_used = branincurrin

if __name__ == "__main__":
    advisor = NSGA2EAdvisor(
        config_space = space,
        task_id='default_task_id',
    )

    used = 1
    MAX_RUNS = 1000
    if used == 0:
        for i in range(MAX_RUNS):
            # ask
            config = advisor.get_suggestion()
            # evaluate
            ret = objfunc_used(config)
            # tell
            observation = Observation(config=config, objs=ret['objs'])
            advisor.update_observation(observation)
            print('===== ITER %d/%d: %s.' % (i+1, MAX_RUNS, observation))
    else:
        MAX_RUNS //= 40
        for i in range(MAX_RUNS):
            # ask
            configs = advisor.get_suggestions()
            observations = []
            # evaluate
            for config in configs:
                ret = objfunc_used(config)
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
