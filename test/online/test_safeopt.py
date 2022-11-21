# License: MIT
import os
import random

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

sys.path.insert(0, ".")

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from openbox.experimental.advanced.safeopt_advisor import SafeOptAdvisor, DefaultBeta, nd_range
from openbox.benchmark.objective_functions.synthetic import Ackley, Rosenbrock, Gaussian, SafetyConstrained, Branin, Bukin
from openbox import Advisor, Observation

try:
    from tqdm import trange
except ModuleNotFoundError:
    trange = range

THRESHOLD = 10

FUNCTIONS = [
    (Ackley(dim=1), 22, 326, 5),
    (Ackley(dim=1), 22, 326, 10),
    (Ackley(dim=1), 22, 326, 15),
    (Ackley(dim=2), 22, 92, 5),
    (Ackley(dim=2), 22, 92, 10),
    (Ackley(dim=2), 22, 92, 15),
    (Branin(), 310, 1580, 200),
    (Branin(), 310, 1580, 50),
    (Branin(), 310, 1580, 20),
    (Bukin(), 230, 1650, 150),
    (Bukin(), 230, 1650, 50),
    (Bukin(), 230, 1650, 20),
    (Gaussian(dim=1), 0.15, 0.41, 0.1),
    (Gaussian(dim=1), 0.15, 0.41, 0.02),
    (Gaussian(dim=1), 0.15, 0.41, 0.005),
    (Gaussian(dim=2), 0.3, 0.5, 0.2),
    (Gaussian(dim=2), 0.3, 0.5, 0.05),
    (Gaussian(dim=2), 0.3, 0.5, 0.01),
    (Rosenbrock(dim=2), 1200000, 6100000, 1000000),
    (Rosenbrock(dim=2), 1200000, 6100000, 10000),
    (Rosenbrock(dim=2), 1200000, 6100000, 100),
]

# Run 5 times for each dataset, and get average value
REPEATS = 2

# The number of function evaluations allowed.
MAX_RUNS = 40

# We need to re-initialize the advisor every time we start a new run.
# So these are functions that provides advisors.
ADVISORS = [
    (lambda sp, r, h, l, s, b: SafeOptAdvisor(config_space=sp, num_constraints=1, random_state=r,
                                              lipschitz=l,
                                              seed_set=s,
                                              sample_size=40000,
                                              beta=DefaultBeta(b=b, sz=40000, delta=0.01),
                                              threshold=h), 'SafeOpt', True),
    (lambda sp, r, h, l, s, b: Advisor(config_space=sp, num_constraints=1, random_state=r, surrogate_type='gp',
                                       acq_type='eic', acq_optimizer_type='random_scipy'), 'GP+EIC', True),
    (lambda sp, r, h, l, s, b: Advisor(config_space=sp, num_constraints=0, random_state=r, surrogate_type='gp',
                                       acq_type='ei', acq_optimizer_type='random_scipy'), 'GP', False),
]

matplotlib.use("Agg")

# Run
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Range')
    parser.add_argument('-f', dest='f', type=int, default=-1)
    parser.add_argument('-t', dest='t', type=int, default=-1)
    parser.add_argument('-af', dest='af', type=int, default=-1)
    parser.add_argument('-at', dest='at', type=int, default=-1)
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

    if args.af == -1 and args.at == -1:
        af = 0
        at = len(ADVISORS)
    elif args.at == -1:
        af = args.af
        at = af + 1
    else:
        af = args.af
        at = args.at

    for function, bound, lipschitz, threshold in FUNCTIONS[f: t]:

        function_name = function.__class__.__name__

        function_name += "({:d})".format(len(function.config_space.keys()))

        function_name += "(h={:d})".format(threshold)

        print("Running dataset " + function_name)

        space = function.config_space

        x0 = space.sample_configuration()
        dim = len(function(x0)['objs'])

        starting_point = None
        starting_value = 1e100

        while True:
            starting_point = space.sample_configuration()
            starting_value = function(starting_point)['objs'][0]
            if starting_value < threshold * 0.75:
                break

        print(starting_point)
        print(starting_value)

        seed_set = np.array([starting_point.get_array()])

        all_results = dict()

        # random_states = list(range(REPEATS))
        random_states = list(random.randint(0, 10000) for i in range(REPEATS))

        for advisor_getter, name, takes_constraint in ADVISORS[af:at]:

            print("Testing Method " + name)

            histories = []
            time_costs = []

            true_f = function if not takes_constraint else SafetyConstrained(function, h=threshold)

            try:
                for r in range(REPEATS):

                    print(f"{r + 1}/{REPEATS}:")

                    start_time = time.time()

                    advisor = advisor_getter(space, random_states[r], threshold, lipschitz, seed_set, bound)

                    for i in trange(MAX_RUNS):
                        config = advisor.get_suggestion()
                        ret = true_f(config)
                        observation = Observation(config=config, objs=ret['objs'],
                                                  constraints=ret[
                                                      'constraints']) if 'constraints' in ret else Observation(
                            config=config, objs=ret['objs'])
                        advisor.update_observation(observation)

                    time_costs.append(time.time() - start_time)
                    histories.append(advisor.get_history())

            except KeyboardInterrupt:
                time_costs.append(time.time() - start_time)
                histories.append(advisor.get_history())

            mins = [[h.perfs[0]] for h in histories]
            minvs = [[h.configurations[0].get_dictionary()] for h in histories]

            alls = [[h.perfs[0]] for h in histories]
            allvs = [[h.configurations[0].get_dictionary()] for h in histories]

            for j, h in enumerate(histories):
                for i in range(1, len(h.perfs)):
                    if h.perfs[i] <= mins[j][-1]:
                        mins[j].append(h.perfs[i])
                        minvs[j].append(h.configurations[i].get_dictionary())
                    else:
                        mins[j].append(mins[j][-1])
                        minvs[j].append(minvs[j][-1])

                    alls[j].append(h.perfs[i])
                    allvs[j].append(h.configurations[i].get_dictionary())

            mean = [np.mean([a[i] for a in mins if i < len(a)]) for i in range(MAX_RUNS)]
            std = [np.std([a[i] for a in mins if i < len(a)]) for i in range(MAX_RUNS)]

            all_results[name] = dict()
            all_results[name]['mean'] = mean
            all_results[name]['std'] = std
            all_results[name]['configs'] = allvs
            all_results[name]['values'] = alls
            all_results[name]['time_costs'] = time_costs
            all_results[name]['random_states'] = random_states

            if name.startswith("SafeOpt") and False:
                advisor: SafeOptAdvisor = advisor

                # advisor.debug(advisor.sets.s_set)
                # advisor.debug()

                plt.cla()
                pts = np.array(
                    list(advisor.sets.get_array(k) for k in nd_range(advisor.sets.size) if advisor.sets.vis_set[k]))

                if pts.shape[1] == 2:
                    plt.scatter(pts[:, 0], pts[:, 1])
                else:
                    plt.scatter(np.arange(pts.shape[0]), pts[:, 0])

                plt.savefig(f"tmp/TMP_EVALS.jpg")

                # ids = np.arange(advisor.sets.size)[advisor.sets.s_set]
                # print(advisor.sets.get_config(np.min(ids)), advisor.sets.get_config(np.max(ids)))

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

        plt.cla()
        for k, v in all_results.items():
            mean = np.array(v['mean'])

            Y = np.array(v['values']).flatten()
            plt.scatter(np.array([np.arange(len(mean)) for i in range(len(v['values']))]).flatten()[:len(Y)],
                        Y, alpha=0.2)

        plt.title(function_name)
        plt.legend()

        plt.savefig(f"tmp/{timestr}_{function_name}_scatter.jpg")
