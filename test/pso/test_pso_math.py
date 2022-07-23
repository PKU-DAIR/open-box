# License: MIT

import numpy as np
import matplotlib.pyplot as plt
from openbox import Advisor, sp, Observation
from openbox.core.pso.pso_advisor import PSOAdvisor
from openbox.core.ea.regularized_ea_advisor import RegularizedEAAdvisor

from openbox.benchmark.objective_functions.synthetic import *

# Define Objective Function
from openbox.utils.config_space import convert_configurations_to_array

alp = np.array([[1.0], [1.2], [3.0], [3.2]])
A = np.array([
    [10, 3, 17, 3.5, 1.7, 8],
    [0.05, 10, 17, 0.1, 8, 14],
    [3, 3.5, 1.7, 10, 17, 8],
    [17, 8, 0.05, 10, 0.1, 14]
])
P = np.array([
    [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
    [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
    [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
    [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]
])


class HARTMANN6(BaseTestProblem):
    def __init__(self, noise_std = 0, random_state = None):
        params = {'x1': (0, 1, 0),
                  'x2': (0, 1, 0),
                  'x3': (0, 1, 0),
                  'x4': (0, 1, 0),
                  'x5': (0, 1, 0),
                  'x6': (0, 1, 0)}
        config_space = ConfigurationSpace()
        config_space.add_hyperparameters([UniformFloatHyperparameter(k, *v) for k, v in params.items()])
        super().__init__(config_space, noise_std,
                         optimal_value = -3.32237,
                         optimal_point = [(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)],
                         random_state = random_state)

    def _evaluate(self, X):
        sum = 0
        for i in range(4):
            tmp = 0
            for j in range(6):
                tmp += A[i][j] * ((X[j] - P[i][j]) ** 2)
            sum += alp[i] * np.exp(-tmp)

        result = dict()
        result['objs'] = [-(2.58 + sum) / 1.94]
        return result


try:
    from tqdm import trange
except ModuleNotFoundError:
    trange = range

function = HARTMANN6()
space = function.config_space
MAX_RUNS = 50
# Use model 0 to test PSOAdvisor, 1 to compare different strategy
TEST_MOD = 2

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
    elif TEST_MOD == 2:
        advisors = [RegularizedEAAdvisor(
            config_space = space,
            task_id = 'default_task_id',
        ),
            PSOAdvisor(
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

    if TEST_MOD == 2:
        pass
        X = []
        Y = []
        res = None
        eaad = advisors[0]
        m = MAX_RUNS * 30
        print("Now running" + str(eaad.__class__))
        for i in trange(m):
            # ask
            config = eaad.get_suggestion()
            # evaluate
            ret = function(config)
            # tell
            observation = Observation(config = config, objs = ret['objs'])
            eaad.update_observation(observation)
            if res is None or res.objs[0] > observation.objs[0]:
                res = observation
            X.append(i)
            Y.append(res.objs[0])
            if trange == range:
                print('===== ITER %d/%d.' % (i + 1, MAX_RUNS))
        print(res)
        plt.plot(X, Y, label = 'Regularized_ea')

        for advisor in advisors[1:]:
            res = None
            X = []
            Y = []
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
                X.append(i * 30)
                Y.append(res.objs[0])
                if trange == range:
                    print('===== ITER %d/%d.' % (i + 1, MAX_RUNS))
            print(res)
            plt.plot(X, Y, label = advisor.w_stg)

    else:
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
