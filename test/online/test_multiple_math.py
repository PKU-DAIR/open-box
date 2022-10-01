# License: MIT
import os
import sys
import time
import argparse

sys.path.insert(0, ".")

from test.test_utils import load_data

from sklearn.metrics import balanced_accuracy_score

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ConfigSpace import Constant, Configuration
from sklearn.model_selection import train_test_split

from openbox.benchmark.objective_functions.synthetic import Ackley, Rosenbrock, Keane
from openbox import Advisor, sp, Observation, get_config_space, get_objective_function

# Define Objective Function
from openbox.core.sync_batch_advisor import SyncBatchAdvisor
from openbox.core.generic_advisor import Advisor
from openbox.core.online.utils.cfo import CFO
from openbox.core.online.utils.flow2 import FLOW2
from openbox.core.online.utils.blendsearch import BlendSearchAdvisor
from openbox.optimizer.generic_smbo import SMBO
from openbox.utils.config_space import convert_configurations_to_array

try:
    from tqdm import trange
except ModuleNotFoundError:
    trange = range

FUNCTIONS = [
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
MAX_RUNS = 200
BATCH_SIZE = 5

# We need to re-initialize the advisor every time we start a new run.
# So these are functions that provides advisors.
ADVISORS = [(lambda sp: BlendSearchAdvisor(globalsearch=Advisor, config_space=sp, task_id='default_task_id'),
             'BlendSearch'),
            (lambda sp: Advisor(config_space=sp), 'SMBO'),
            (lambda sp: SyncBatchAdvisor(config_space=sp, batch_size=BATCH_SIZE), 'BatchBO')]

matplotlib.use("Agg")

# Run
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Range')
    parser.add_argument('--f', dest='f', type=int, default=0)
    parser.add_argument('--t', dest='t', type=int, default=0)
    args = parser.parse_args()

    for function in FUNCTIONS[args.f:(len(FUNCTIONS) if args.t == 0 else args.t)]:

        function_name = function.__class__.__name__

        if hasattr(function, "dim"):
            function_name += "({:d})".format(function.dim)

        print("Running dataset " + function_name)

        space = function.config_space

        x0 = space.sample_configuration()

        dim = len(function(x0)['objs'])

        avg_results = {}

        for advisor_getter, name in ADVISORS:

            print("Testing Method " + name)

            histories = []

            for r in range(REPEATS):

                print(f"{r + 1}/{REPEATS}:")

                advisor = advisor_getter(space)

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

                histories.append(advisor.get_history())

            mins = [[h.perfs[0]] for h in histories]

            for i in range(1, MAX_RUNS):
                for j, h in enumerate(histories):
                    mins[j].append(min(mins[j][-1], h.perfs[i]))

            fmins = [sum(a[i] for a in mins) / REPEATS for i in range(MAX_RUNS)]

            avg_results[name] = fmins

        timestr = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())

        if not os.path.exists("tmp"):
            os.mkdir("tmp")

        with open(f"tmp/{timestr}_{function_name}.txt", "w") as f:
            f.write(str(avg_results))

        plt.cla()
        for k, v in avg_results.items():
            plt.plot(v, label=k)

        plt.title(function_name)
        plt.legend()

        plt.savefig(f"tmp/{timestr}_{function_name}.jpg")
