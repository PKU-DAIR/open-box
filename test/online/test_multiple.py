# License: MIT
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from ConfigSpace import Constant
from sklearn.model_selection import train_test_split

from openbox import Advisor, sp, Observation, get_config_space, get_objective_function

from openbox.benchmark.objective_functions.synthetic import Bukin
from openbox.benchmark.objective_functions.synthetic import Ackley

import openml

# Define Objective Function
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

DATASETS = ['phoneme', 'ozone-level-8hr', 'scene', 'musk', 'gina_agnostic', 'optdigits', 'abalone', 'satimage',
            'SpeedDating', 'first-order-theorem-proving']

# Run 5 times for each dataset, and get average value
REPEATS = 5

# The number of function evaluations allowed.
MAX_RUNS = 200

# We need to re-initialize the advisor every time we start a new run.
# So these are functions that provides advisors.
ADVISORS = [lambda sp: BlendSearchAdvisor(globalsearch=FLOW2, config_space=sp, task_id='default_task_id'),
            lambda sp: BlendSearchAdvisor(globalsearch=Advisor, config_space=sp, task_id='default_task_id')]

# Run
if __name__ == "__main__":

    for dataset_name in DATASETS:

        print("Running dataset " + dataset_name)

        dataset = openml.datasets.get_dataset(dataset_name)

        Xy, _, classes, names = dataset.get_data(dataset_format='array')

        X, y = Xy[:, :-1], Xy[:, -1]
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

        space = get_config_space('lightgbm')
        space.add_hyperparameters([Constant('n_jobs', 5)])

        function = get_objective_function('lightgbm', x_train, x_val, y_train, y_val)

        x0 = space.sample_configuration()

        dim = len(function(x0)['objs'])

        avg_results = {}

        for advisor_getter in ADVISORS:

            advisor = advisor_getter(space)
            name = str(advisor)

            print("Testing Method " + name)

            histories = []

            for r in range(REPEATS):

                print(f"{r + 1}/{REPEATS}:")

                advisor = advisor_getter(space)

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

        with open(f"tmp/{timestr}_{dataset_name}.txt", "w") as f:
            f.write(str(avg_results))

        plt.cla()
        for k, v in avg_results.items():
            plt.plot(v, label=k)

        plt.title(dataset_name)
        plt.legend()

        plt.savefig(f"tmp/{timestr}_{dataset_name}.jpg")
