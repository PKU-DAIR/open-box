# License: MIT

import numpy as np
import matplotlib.pyplot as plt
from openbox import Advisor, sp, Observation
from openbox.core.ea.differential_ea_advisor import DifferentialEAAdvisor
from openbox.core.ea.regularized_ea_advisor import RegularizedEAAdvisor
from openbox.core.ea.adaptive_ea_advisor import AdaptiveEAAdvisor
from openbox.core.ea.cmaes_ea_advisor import CMAESEAAdvisor
from openbox.core.ea.nsga2_ea_advisor import NSGA2EAdvisor

from openbox.benchmark.objective_functions.synthetic import *

# Define Objective Function
from openbox.core.ea.saea_advisor import SAEA_Advisor

try:
    from tqdm import trange
except ModuleNotFoundError:
    trange = range


function = DTLZ2(num_objs=2,constrained=True)
space = function.config_space


# Run
if __name__ == "__main__":
    advisors = [CMAESEAAdvisor(
        config_space = space,
        num_objs=2,
        task_id='default_task_id',
    ),SAEA_Advisor(
        config_space = space,
        num_objs=2,
        task_id='default_task_id',
    ),DifferentialEAAdvisor(
        config_space = space,
        num_objs=2,
        task_id='default_task_id',
    ),NSGA2EAdvisor(
        config_space = space,
        num_objs=2,
        task_id='default_task_id',
    ),AdaptiveEAAdvisor(
        config_space = space,
        num_objs=2,
        task_id='default_task_id',
    )]

    res = function(space.sample_configuration())
    dim = len(res['objs'])


    axes = None
    histories = []

    MAX_RUNS = 2000
    for advisor in advisors:
        print("Now running" + str(advisor.__class__))
        m = MAX_RUNS // 8 if isinstance(advisor,AdaptiveEAAdvisor) else MAX_RUNS

        for i in trange(m):
            # ask
            config = advisor.get_suggestion()
            # evaluate
            ret = function(config)
            # tell
            observation = Observation(config=config, objs=ret['objs'])
            advisor.update_observation(observation)
            if trange == range:
                print('===== ITER %d/%d.' % (i+1, MAX_RUNS))


        history = advisor.get_history()
        histories.append(history.get_incumbents())

        if dim == 1:
            axes = history.plot_convergence(ax=axes)
        elif dim == 2:
            inc = history.get_incumbents()
            inc.sort(key=lambda x: x[1][0])
            plt.plot([x[1][0] for x in inc],[x[1][1] for x in inc],label = advisor.__class__.__name__)

    if dim <= 2:
        plt.legend()
        plt.show()

    for i,h in enumerate(histories):
        print(advisors[i].__class__)
        print(h)


    # install pyrfr to use get_importance()
    # print(history.get_importance())

    # history.visualize_jupyter()
