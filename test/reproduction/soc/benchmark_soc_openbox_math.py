"""
example cmdline:

python test/optimizer/so/benchmark_so_openbox_math.py --problem branin --n 200 --init 3 --surrogate gp --optimizer scipy --rep 1 --start_id 0

"""
import os
import sys
import time
import numpy as np
import argparse
import pickle as pkl

sys.path.insert(0, os.getcwd())
from test.reproduction.soc.soc_benchmark_function import get_problem
from openbox.optimizer.generic_smbo import SMBO
from openbox.test_utils import timeit, seeds

parser = argparse.ArgumentParser()
parser.add_argument('--problem', type=str, default='townsend')
parser.add_argument('--n', type=int, default=200)
parser.add_argument('--nc', type=int, default=1)
parser.add_argument('--init', type=int, default=3)
parser.add_argument('--init_strategy', type=str, default='random_explore_first')
parser.add_argument('--surrogate', type=str, default='gp', choices=['gp'])
parser.add_argument('--optimizer', type=str, default='scipy', choices=['scipy', 'local'])
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)

args = parser.parse_args()
problem_str = args.problem
max_runs = args.n
num_constraints = args.nc
initial_runs = args.init
init_strategy = args.init_strategy
surrogate_type = args.surrogate
if args.optimizer == 'scipy':
    acq_optimizer_type = 'random_scipy'
elif args.optimizer == 'local':
    acq_optimizer_type = 'local_random'
else:
    raise ValueError('Unknown optimizer %s' % args.optimizer)
rep = args.rep
start_id = args.start_id
mth = 'openbox'

problem = get_problem(problem_str)
cs = problem.get_configspace(optimizer='smac')
time_limit_per_trial = 600
task_id = '%s_%s' % (mth, problem_str)


def evaluate(mth, run_i, seed):
    print(mth, run_i, seed, '===== start =====', flush=True)

    def objective_function(config):
        y = problem.evaluate_config(config)
        return y

    bo = SMBO(objective_function, cs,
              num_constraints=num_constraints,
              surrogate_type=surrogate_type,  # default: gp
              acq_optimizer_type=acq_optimizer_type,  # default: random_scipy
              initial_runs=initial_runs,  # default: 3
              init_strategy=init_strategy,  # default: random_explore_first
              max_runs=max_runs + initial_runs,
              time_limit_per_trial=time_limit_per_trial, task_id=task_id, random_state=seed)
    # bo.run()
    config_list = []
    perf_list = []
    time_list = []
    global_start_time = time.time()
    for i in range(max_runs):
        config, trial_state, constraints, objs = bo.iterate()
        global_time = time.time() - global_start_time
        origin_perf = objs[0]
        if any(c > 0 for c in constraints):
            perf = 9999999.0
        else:
            perf = origin_perf
        print(seed, i, perf, config, constraints, trial_state, 'time=', global_time)
        config_list.append(config)
        perf_list.append(perf)
        time_list.append(global_time)

    return config_list, perf_list, time_list


with timeit('%s all' % (mth,)):
    for run_i in range(start_id, start_id + rep):
        seed = seeds[run_i]
        with timeit('%s %d %d' % (mth, run_i, seed)):
            # Evaluate
            config_list, perf_list, time_list = evaluate(mth, run_i, seed)

            # Save result
            print('=' * 20)
            print(seed, mth, config_list, perf_list, time_list)
            print(seed, mth, 'best perf', np.min(perf_list))

            timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            dir_path = 'logs/soc_benchmark_%s_%d/%s/' % (problem_str, max_runs, mth)
            file = 'benchmark_%s_%04d_%s.pkl' % (mth, seed, timestamp)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            with open(os.path.join(dir_path, file), 'wb') as f:
                save_item = (config_list, perf_list, time_list)
                pkl.dump(save_item, f)
            print(dir_path, file, 'saved!', flush=True)
