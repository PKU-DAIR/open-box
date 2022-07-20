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
MAX_RUNS = 100
# Use model 0 to test PSOAdvisor, 1 to compare different strategy
TEST_MOD = 0

# Run
if __name__ == "__main__":
    advisors = []
    if TEST_MOD == 0:
        advisors = [PSOAdvisor(
                config_space = space,
                w_stg = 'default',
                task_id = 'default_task_id',
            )]
    elif TEST_MOD == 1:
        advisors = [PSOAdvisor(
                config_space = space,
                w_stg = 'default',
                task_id = 'default_task_id',
            ),
            PSOAdvisor(
                config_space = space,
                w_stg = 'dec',
                max_iter = MAX_RUNS,
                task_id = 'default_task_id',
            ),
            PSOAdvisor(
                config_space = space,
                w_stg = 'rand',
                max_iter = MAX_RUNS,
                task_id = 'default_task_id',
            )]

    res = function(space.sample_configuration())
    dim = len(res['objs'])

    for advisor in advisors:
        res = None
        X_all = []
        Y_all = []
        Y_cur = []
        X_best = []
        Y_best = []
        history = []
        print("Now running" + str(advisor.__class__))
        m = MAX_RUNS

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
        if len(advisors) == 1:
            plt.scatter(X_all, Y_all, alpha = 0.1)
            plt.plot(X_best, Y_cur, c = 'green', label = 'every iter')
            plt.plot(X_best, Y_best, c = 'orange', label = 'best result')
        else:
            plt.plot(X_best, Y_cur, label = advisor.w_stg)

    plt.legend()
    plt.show()
