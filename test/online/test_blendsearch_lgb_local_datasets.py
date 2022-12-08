# License: MIT
import os
import sys
import time

sys.path.insert(0, ".")

from test.test_utils import load_data

from sklearn.metrics import balanced_accuracy_score

import matplotlib
import matplotlib.pyplot as plt
from ConfigSpace import Configuration
from sklearn.model_selection import train_test_split

from openbox import Observation, get_config_space

# Define Objective Function
from openbox.core.sync_batch_advisor import SyncBatchAdvisor
from openbox.core.generic_advisor import Advisor
from openbox.experimental.online.blendsearch import BlendSearchAdvisor

try:
    from tqdm import trange
except ModuleNotFoundError:
    trange = range


def lgbm_function(x_train, x_val, y_train, y_val, task_type = 'cls'):
    from lightgbm import LGBMClassifier

    def cls_objective_function(config: Configuration):
        # convert Configuration to dict
        params = config.get_dictionary()

        # fit model
        model = LGBMClassifier(**params, n_jobs = 6)
        model.fit(x_train, y_train)

        # predict and calculate loss
        y_pred = model.predict(x_val)
        loss = 1 - balanced_accuracy_score(y_val, y_pred)  # OpenBox minimizes the objective

        # return result dictionary
        result = dict(objs = (loss,))
        return result

    if task_type == 'cls':
        objective_function = cls_objective_function
    elif task_type == 'rgs':
        raise NotImplementedError
    else:
        raise ValueError('Unsupported task type: %s.' % (task_type,))
    return objective_function


DATASETS = ['puma8NH', 'wind']

# Run 5 times for each dataset, and get average value
REPEATS = 5

# The number of function evaluations allowed.
MAX_RUNS = 5
BATCH_SIZE = 5

# We need to re-initialize the advisor every time we start a new run.
# So these are functions that provides advisors.
ADVISORS = [(lambda sp: BlendSearchAdvisor(globalsearch = Advisor, config_space = sp, task_id = 'OpenBox'),
             'BlendSearch'),
            (lambda sp: Advisor(config_space = sp), 'SMBO'),
            (lambda sp: SyncBatchAdvisor(config_space = sp, batch_size = BATCH_SIZE), 'BatchBO')]

matplotlib.use("Agg")

# Run
if __name__ == "__main__":

    for dataset_name in DATASETS[1:2]:

        print("Running dataset " + dataset_name)

        # X, y = load_data(dataset_name, "/root/ezio/cls_datasets")
        X, y = load_data(dataset_name, "~/Desktop/cls_datasets")
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 1)

        space = get_config_space('lightgbm')

        function = lgbm_function(x_train, x_val, y_train, y_val)

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
                            observation = Observation(config = config, objs = ret['objs'])
                            advisor.update_observation(observation)
                        if trange == range:
                            print('===== ITER %d/%d.' % ((i + 1) * BATCH_SIZE, MAX_RUNS))
                else:
                    for i in trange(MAX_RUNS):
                        config = advisor.get_suggestion()
                        ret = function(config)
                        observation = Observation(config = config, objs = ret['objs'])
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
            plt.plot(v, label = k)

        plt.title(dataset_name)
        plt.legend()

        plt.savefig(f"tmp/{timestr}_{dataset_name}.jpg")
