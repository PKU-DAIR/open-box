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

sys.path.insert(0, ".")

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from ConfigSpace import Configuration

# from openbox.core.highdim.safeopt_advisor import SafeOptAdvisor
# from openbox.core.highdim.turbo_advisor import TuRBOAdvisor
from openbox.benchmark.objective_functions.synthetic import Ackley
from openbox import Observation
from openbox.experimental.highdim.linebo_advisor import LineBOAdvisor

try:
    from tqdm import trange
except ModuleNotFoundError:
    trange = range

FUNCTIONS = [
    Ackley(dim=3),
]

# Run 5 times for each dataset, and get average value
REPEATS = 5

# The number of function evaluations allowed.
MAX_RUNS = 50
BATCH_SIZE = 5

# We need to re-initialize the advisor every time we start a new run.
# So these are functions that provides advisors.
ADVISORS = [
    (lambda sp, r: LineBOAdvisor(config_space=sp, random_state=r), 'LineBO'),
    # (lambda sp, r: BlendSearchAdvisor(
    #     globalsearch=(Advisor, tuple(), dict(surrogate_type='gp', acq_type='ei', acq_optimizer_type='random_scipy')),
    #     config_space=sp, random_state=r), 'BlendSearch'),
    # (lambda sp, r: SyncBatchAdvisor(config_space=sp, surrogate_type='gp', acq_type='ei',
    #                                 acq_optimizer_type='random_scipy', batch_size=BATCH_SIZE, random_state=r),
    #  'BatchBO'),
    # (lambda sp, r: Advisor(config_space=sp, random_state=r), 'BO(Default)'),
    # (lambda sp, r: Advisor(config_space=sp, surrogate_type='gp', acq_type='ei', acq_optimizer_type='random_scipy',
    #                       random_state=r), 'BO(GP+RandomScipy)'),

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

    for function in FUNCTIONS[f: t]:

        function_name = function.__class__.__name__

        function_name += "({:d})".format(len(function.config_space.keys()))

        print("Running dataset " + function_name)

        space = function.config_space

        x0 = space.sample_configuration()

        dim = len(function(x0)['objs'])

        all_results = dict()

        random_states = list(range(REPEATS))

        for advisor_getter, name in ADVISORS[af:at]:

            print("Testing Method " + name)

            histories = []
            time_costs = []

            try:
                for r in range(REPEATS):

                    print(f"{r + 1}/{REPEATS}:")

                    start_time = time.time()

                    advisor = advisor_getter(space, random_states[r])

                    if name == 'BatchBO' or name == 'TuRBO':
                        for i in trange(MAX_RUNS // BATCH_SIZE):
                            configs = advisor.get_suggestions(batch_size=BATCH_SIZE)
                            # print(len(configs))
                            for config in configs:
                                ret = function(config)
                                observation = Observation(config=config, objs=ret['objs'])
                                advisor.update_observation(observation)
                            if trange == range:
                                print('===== ITER %d/%d.' % ((i + 1) * BATCH_SIZE, MAX_RUNS))
                    else:
                        for i in trange(MAX_RUNS):
                            config = advisor.get_suggestion()
                            # print(config.get_array())
                            # print("function return ", config.get_array())
                            ret = function(config)
                            observation = Observation(config=config, objs=ret['objs'])
                            advisor.update_observation(observation)
                            # print("result ", ret['objs'])
                            if trange == range:
                                print('===== ITER %d/%d.' % (i + 1, MAX_RUNS))

                    time_costs.append(time.time() - start_time)
                    histories.append(advisor.get_history())

            except KeyboardInterrupt:
                time_costs.append(time.time() - start_time)
                histories.append(advisor.get_history())

            if True and (name[:6] == "LineBO"):
                print("Plotting history evaluations:")
                plt.cla()
                plt.scatter(np.array([i['x1'] for i in advisor.get_history().configurations]),
                            np.array([i['x2'] for i in advisor.get_history().configurations]),
                            c=np.array([(i.get_array()[2], 0, 1 - i.get_array()[2]) for i in
                                        advisor.get_history().configurations]) if dim >= 3 else 'b')

                for x0, x1 in advisor.history_lines:
                    c0 = Configuration(space, vector=x0)
                    c1 = Configuration(space, vector=x1)
                    plt.plot([c0['x1'], c1['x1']], [c0['x2'], c1['x2']])

                if not os.path.exists("tmp"):
                    os.mkdir("tmp")

                plt.savefig(f"tmp/TMP.jpg")

                print("Testing GP accuracy:")

                # print(advisor.last_gp_data)

                for i in range(50):
                    config = space.sample_configuration()

                    config = advisor.to_original_space(advisor.line_space.sample_configuration())
                    # config = Configuration(space, vector = advisor.last_gp_data[0][i] + np.random.randn(12) * 0.01)
                    res1 = function(config)['objs'][0]
                    res2 = advisor.objective_surrogate.predict(np.array([config.get_array()]))
                    print(config.get_array())
                    print(
                        f"Sampled point at [{config['x1']},{config['x2']}], function return {res1}, gp return {res2}")

            mins = [[h.perfs[0]] for h in histories]
            minvs = [[h.configurations[0].get_dictionary()] for h in histories]

            for j, h in enumerate(histories):
                for i in range(1, len(h.perfs)):
                    if h.perfs[i] <= mins[j][-1]:
                        mins[j].append(h.perfs[i])
                        minvs[j].append(h.configurations[i].get_dictionary())
                    else:
                        mins[j].append(mins[j][-1])
                        minvs[j].append(minvs[j][-1])

            mean = [np.mean([a[i] for a in mins if i < len(a)]) for i in range(MAX_RUNS)]
            std = [np.std([a[i] for a in mins if i < len(a)]) for i in range(MAX_RUNS)]

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
            plt.plot(mean, scaley='log', label=k)
            plt.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.2)

        plt.title(function_name)
        plt.legend()

        plt.savefig(f"tmp/{timestr}_{function_name}.jpg")
