# License: MIT

import numpy as np
import matplotlib.pyplot as plt
from openbox import Advisor, sp, Observation
from openbox.core.pso.pso_advisor import PSOAdvisor

from openbox.benchmark.objective_functions.synthetic import *

# Define Objective Function
from openbox.utils.config_space import convert_configurations_to_array

try:
    from tqdm import trange
except ModuleNotFoundError:
    trange = range

function = Branin()
space = function.config_space

# Run
if __name__ == "__main__":
    advisors = [PSOAdvisor(
        config_space = space,
        task_id = 'default_task_id',
    )]

    res = function(space.sample_configuration())
    dim = len(res['objs'])

    axes = None
    X_all = []
    Y_all = []
    Y_cur = []
    X_best = []
    Y_best = []
    history = []
    res = None

    MAX_RUNS = 100
    for advisor in advisors:
        print("Now running" + str(advisor.__class__))
        m = MAX_RUNS

        pdy = []

        for i in trange(m):
            # ask
            configs = advisor.get_suggestions()
            # evaluate
            observations = []
            rets = []
            for config in configs:
                ret = function(config)
                rets.append(ret)
                tmp = Observation(config = config, objs = ret['objs'])
                observations.append(tmp)
                if res is None or res.objs[0] > tmp.objs[0]:
                    res = tmp
            # tell
            advisor.update_observations(observations)

            history += observations

            kk = []
            for t in observations:
                X_all.append(i)
                Y_all.append(t.objs[0])
                kk.append(t.objs[0])
            X_best.append(i)
            Y_cur.append(min(kk))
            Y_best.append(res.objs[0])

            if trange == range:
                print('===== ITER %d/%d.' % (i + 1, MAX_RUNS))

        print(res)
        plt.scatter(X_all, Y_all, alpha = 0.1)
        plt.plot(X_best, Y_cur, c = 'green', label = 'every iter')
        plt.plot(X_best, Y_best, c = 'orange', label = 'best result')
        plt.legend()
        plt.show()
