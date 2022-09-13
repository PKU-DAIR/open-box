# License: MIT

import numpy as np
import matplotlib.pyplot as plt
from openbox import Advisor, sp, Observation
from openbox.core.ea.differential_ea_advisor import DifferentialEAAdvisor
from openbox.core.ea.regularized_ea_advisor import RegularizedEAAdvisor
from openbox.core.ea.adaptive_ea_advisor import AdaptiveEAAdvisor
from openbox.core.ea.cmaes_ea_advisor import CMAESEAAdvisor
from openbox.core.ea.nsga2_ea_advisor import NSGA2EAdvisor

from openbox.benchmark.objective_functions.synthetic import Rosenbrock

# Define Objective Function
from openbox.core.ea.saea_advisor import SAEAAdvisor
from openbox.core.online.utils.cfo import CFO
from openbox.core.online.utils.flow2 import FLOW2
from openbox.utils.config_space import convert_configurations_to_array

try:
    from tqdm import trange
except ModuleNotFoundError:
    trange = range

function = Rosenbrock()
space = function.config_space

x0 = space.sample_configuration()

# Run
if __name__ == "__main__":
    advisors = [FLOW2(
        config_space=space,
        task_id='default_task_id',
        x0=x0
    ), CFO(
        config_space=space,
        task_id='default_task_id',
        x0=x0
    )]

    res = function(space.sample_configuration())
    dim = len(res['objs'])

    axes = None
    histories = []

    MAX_RUNS = 1000
    for advisor in advisors:
        print("Now running" + str(advisor.__class__))
        m = MAX_RUNS

        for i in trange(m):
            # ask
            config = advisor.get_suggestion()
            # print(config)
            # evaluate
            ret = function(config)
            # tell
            observation = Observation(config=config, objs=ret['objs'])
            advisor.update_observation(observation)

            if trange == range:
                print('===== ITER %d/%d.' % (i + 1, MAX_RUNS))

        history = advisor.get_history()
        histories.append(history.get_incumbents())

        if dim == 1:
            axes = history.plot_convergence(ax=axes, yscale='log', name = advisor.__class__.__name__)
        elif dim == 2:
            inc = history.get_incumbents()
            inc.sort(key=lambda x: x[1][0])
            plt.plot([x[1][0] for x in inc], [x[1][1] for x in inc])

    if dim <= 2:
        plt.legend()
        plt.show()

    for i, h in enumerate(histories):
        print(advisors[i].__class__.__name__)
        print(h)
