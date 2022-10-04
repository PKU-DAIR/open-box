# License: MIT
import os

NUM_THREADS = "1"
os.environ["OMP_NUM_THREADS"] = NUM_THREADS  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = NUM_THREADS  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS  # export NUMEXPR_NUM_THREADS=1

import sys
import time
import argparse
import json

from openbox.core.highdim.linebo_advisor import LineBOAdvisor

sys.path.insert(0, ".")

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from ConfigSpace import Constant, Configuration, UniformFloatHyperparameter, ConfigurationSpace

from openbox.benchmark.objective_functions.synthetic import Ackley, Rosenbrock, Keane, BaseTestProblem
from openbox import Advisor, sp, Observation, get_config_space, get_objective_function

# Define Objective Function
from openbox.core.sync_batch_advisor import SyncBatchAdvisor
from openbox.core.generic_advisor import Advisor
from openbox.core.online.utils.blendsearch import BlendSearchAdvisor

try:
    from tqdm import trange
except ModuleNotFoundError:
    trange = range


class Schwefel(BaseTestProblem):
    r"""Generalized Schwefel's Problem 2.26.

    d-dimensional function (usually evaluated on `[-500, 500]^d`):

        f(x) = \sum_{i=1}^{d} -x_i sin(\sqrt{\abs{x_i}})

    """

    def __init__(self, dim=2, noise_std=0, random_state=None):
        self.dim = dim
        params = {'x%d' % i: (-500.0, 500.0, 100.0) for i in range(1, 1 + self.dim)}
        config_space = ConfigurationSpace()
        config_space.add_hyperparameters([UniformFloatHyperparameter(k, *v) for k, v in params.items()])
        super().__init__(config_space, noise_std,
                         optimal_value=-12596.5,
                         random_state=random_state)

    def _evaluate(self, X):
        result = dict()
        result['objs'] = [np.sum(-X * np.sin(np.sqrt(np.abs(X))), axis=-1)]
        return result


class Rastrigin(BaseTestProblem):
    r"""Generalized Rastrigin's Function

    d-dimensional function (usually evaluated on `[-5.12, 5.12]^d`):

        f(x) = \sum_{i=1}^{d} (x_i^2 - 10 \cos(2\pi x) + 10)

    """

    def __init__(self, dim=2, noise_std=0, random_state=None):
        self.dim = dim
        params = {'x%d' % i: (-5.12, 5.12, 1) for i in range(1, 1 + self.dim)}
        config_space = ConfigurationSpace()
        config_space.add_hyperparameters([UniformFloatHyperparameter(k, *v) for k, v in params.items()])
        super().__init__(config_space, noise_std,
                         optimal_value=0,
                         random_state=random_state)

    def _evaluate(self, X):
        result = dict()
        result['objs'] = [np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X) + 10, axis=-1)]
        return result


class Griewank(BaseTestProblem):
    r"""Generalized Griewank's Function

    d-dimensional function (usually evaluated on `[-600, 600]^d`):

        f(x) = 1/4000 \sum_{i=1}^{d} - \prod_{i=1}^d \cos(x_i / \sqrt{i}) + 1

    """

    def __init__(self, dim=2, noise_std=0, random_state=None):
        self.dim = dim
        params = {'x%d' % i: (-600, 600, 100) for i in range(1, 1 + self.dim)}
        config_space = ConfigurationSpace()
        config_space.add_hyperparameters([UniformFloatHyperparameter(k, *v) for k, v in params.items()])
        super().__init__(config_space, noise_std,
                         optimal_value=0,
                         random_state=random_state)

    def _evaluate(self, X):
        result = dict()
        result['objs'] = [np.sum(X ** 2, axis=-1) / 4000.0 -
                          np.prod(np.cos(X / (np.arange(1, self.dim + 1)) ** 0.5)) + 1]
        return result


FUNCTIONS = [
    Schwefel(dim=12),
    Schwefel(dim=15),
    Schwefel(dim=20),
    Schwefel(dim=25),
    Schwefel(dim=30),
    Schwefel(dim=40),
    Rastrigin(dim=12),
    Rastrigin(dim=15),
    Rastrigin(dim=20),
    Rastrigin(dim=25),
    Rastrigin(dim=30),
    Rastrigin(dim=40),
    Ackley(dim=12),
    Ackley(dim=15),
    Ackley(dim=20),
    Ackley(dim=25),
    Ackley(dim=30),
    Ackley(dim=40),
    Rosenbrock(dim=12),
    Rosenbrock(dim=15),
    Rosenbrock(dim=20),
    Rosenbrock(dim=25),
    Rosenbrock(dim=30),
    Rosenbrock(dim=40),
    Keane(dim=12),
    Keane(dim=15),
    Keane(dim=20),
    Keane(dim=25),
    Keane(dim=30),
    Keane(dim=40),
]

# Run 5 times for each dataset, and get average value
REPEATS = 5

# The number of function evaluations allowed.
MAX_RUNS = 500
BATCH_SIZE = 10

# We need to re-initialize the advisor every time we start a new run.
# So these are functions that provides advisors.
ADVISORS = [
    (lambda sp, r: LineBOAdvisor(config_space=sp, random_state=r), 'LineBO'),
    (lambda sp, r: BlendSearchAdvisor(globalsearch=Advisor, config_space=sp, random_state=r), 'BlendSearch'),
    (lambda sp, r: SyncBatchAdvisor(config_space=sp, batch_size=BATCH_SIZE, random_state=r), 'BatchBO'),
    (lambda sp, r: Advisor(config_space=sp, random_state=r), 'BO(Default)'),
    (lambda sp, r: Advisor(config_space=sp, surrogate_type='gp', acq_type='ei', acq_optimizer_type='random_scipy',
                           random_state=r), 'BO(GP+RandomScipy)'),

]

matplotlib.use("Agg")

# Run
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Range')
    parser.add_argument('-f', dest='f', type=int, default=-1)
    parser.add_argument('-t', dest='t', type=int, default=-1)
    args = parser.parse_args()

    if args.f == -1 and args.t == -1:
        f = 0
        t = len(FUNCTIONS)
    elif args.t == -1:
        f = args.f
        t = f + 1
    else:
        f = args.f
        t = args.t

    for function in FUNCTIONS[f: t]:

        function_name = function.__class__.__name__

        function_name += "({:d})".format(len(function.config_space.keys()))

        print("Running dataset " + function_name)

        space = function.config_space

        x0 = space.sample_configuration()

        dim = len(function(x0)['objs'])

        all_results = dict()

        random_states = list(range(REPEATS))

        for advisor_getter, name in ADVISORS:

            print("Testing Method " + name)

            histories = []
            time_costs = []

            for r in range(REPEATS):

                print(f"{r + 1}/{REPEATS}:")

                start_time = time.time()

                advisor = advisor_getter(space, random_states[r])

                if name == 'BatchBO':
                    for i in trange(MAX_RUNS // BATCH_SIZE):
                        configs = advisor.get_suggestions()
                        for config in configs:
                            ret = function(config)
                            observation = Observation(config=config, objs=ret['objs'])
                            advisor.update_observation(observation)
                        if trange == range:
                            print('===== ITER %d/%d.' % ((i + 1) * BATCH_SIZE, MAX_RUNS))
                else:
                    for i in trange(MAX_RUNS):
                        config = advisor.get_suggestion()
                        ret = function(config)
                        observation = Observation(config=config, objs=ret['objs'])
                        advisor.update_observation(observation)
                        if trange == range:
                            print('===== ITER %d/%d.' % (i + 1, MAX_RUNS))

                time_costs.append(time.time() - start_time)
                histories.append(advisor.get_history())

            mins = [[h.perfs[0]] for h in histories]
            minvs = [[h.configurations[0].get_dictionary()] for h in histories]

            for i in range(1, MAX_RUNS):
                for j, h in enumerate(histories):
                    if h.perfs[i] <= mins[j][-1]:
                        mins[j].append(h.perfs[i])
                        minvs[j].append(h.configurations[i].get_dictionary())
                    else:
                        mins[j].append(mins[j][-1])
                        minvs[j].append(minvs[j][-1])

            mean = [np.mean([a[i] for a in mins]) for i in range(MAX_RUNS)]
            std = [np.std([a[i] for a in mins]) for i in range(MAX_RUNS)]

            all_results[name] = dict()
            all_results[name]['mean'] = mean
            all_results[name]['std'] = std
            all_results[name]['configs'] = minvs
            all_results[name]['values'] = mins
            all_results[name]['time_costs'] = time_costs
            all_results[name]['random_states'] = random_states

        timestr = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())

        if not os.path.exists("tmp"):
            os.mkdir("tmp")

        with open(f"tmp/{timestr}_{function_name}.txt", "w") as f:
            f.write(json.dumps(all_results))

        plt.cla()
        for k, v in all_results.items():
            mean = np.array(v['mean'])
            std = np.array(v['std'])
            plt.plot(mean, label=k)
            plt.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.2)

        plt.title(function_name)
        plt.legend()

        plt.savefig(f"tmp/{timestr}_{function_name}.jpg")
