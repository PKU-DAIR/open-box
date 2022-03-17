"""
example cmdline:

python test/surrogate/benchmark_math.py --problem branin --n 200 --surrogate gp --optimizer local --rep 1 --start_id 0

"""
import os
import sys
import time
import numpy as np
import argparse
import pickle as pkl

sys.path.insert(0, os.getcwd())
from test.reproduction.so.so_benchmark_function import get_problem
from openbox.optimizer.generic_smbo import SMBO
from test.reproduction.test_utils import timeit, seeds

parser = argparse.ArgumentParser()
parser.add_argument('--problem', type=str)      # ackley-16, ackley-32, branin, beale, hartmann
parser.add_argument('--n', type=int, default=200)
parser.add_argument('--init', type=int, default=3)
parser.add_argument('--init_strategy', type=str, default='random_explore_first')
parser.add_argument('--surrogate', type=str, default='gp', choices=['gp', 'gp_mcmc', 'prf', 'lightgbm', 'tpe', 'sk_prf'])
parser.add_argument('--optimizer', type=str, default='local', choices=['scipy', 'local'])
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--use_rand_seed', type=int, default=1)   # use random time seed instead of seeds

parser.add_argument('--tpe_num_samples', type=int, default=64)

args = parser.parse_args()
problem_str = args.problem
max_runs = args.n
initial_runs = args.init
init_strategy = args.init_strategy
surrogate_type = args.surrogate
if surrogate_type == 'tpe':
    advisor_type = 'tpe'
else:
    advisor_type = 'default'
if args.optimizer == 'scipy':
    acq_optimizer_type = 'random_scipy'
elif args.optimizer == 'local':
    acq_optimizer_type = 'local_random'
else:
    raise ValueError('Unknown optimizer %s' % args.optimizer)
rep = args.rep
start_id = args.start_id
use_rand_seed = args.use_rand_seed

tpe_num_samples = args.tpe_num_samples

mth = surrogate_type

problem = get_problem(problem_str)
try:
    problem.n_jobs = 8
except Exception:
    pass

cs = problem.get_configspace(optimizer='smac')
time_limit_per_trial = 3600


def evaluate(mth, run_i, seed):
    print(mth, run_i, seed, '===== start =====', flush=True)

    def objective_function(config):
        y = problem.evaluate_config(config)
        res = dict()
        # res['config'] = config
        res['objs'] = (y,)
        # res['constraints'] = None
        return res

    task_id = '%s_%s_%d' % (mth, problem_str, seed)
    bo = SMBO(objective_function, cs,
              advisor_type=advisor_type,                # choices: default, tpe
              surrogate_type=surrogate_type,            # choices: gp, gp_mcmc, prf, lightgbm
              acq_optimizer_type=acq_optimizer_type,    # default: local_random
              initial_runs=initial_runs,                # default: 3
              init_strategy=init_strategy,              # default: random_explore_first
              max_runs=max_runs,
              time_limit_per_trial=time_limit_per_trial, task_id=task_id, random_state=seed)
    if advisor_type == 'tpe':
        bo.config_advisor.num_samples = tpe_num_samples

    bo.run()
    config_list = bo.get_history().configurations
    perf_list = bo.get_history().perfs
    time_list = bo.get_history().update_times

    return config_list, perf_list, time_list


with timeit('%s all' % (mth,)):
    for run_i in range(start_id, start_id + rep):
        if use_rand_seed == 1:
            base_seed = int(time.time() * 10000 % 10000)
            rng = np.random.RandomState(base_seed)
            seed = rng.randint(10000)
        else:
            seed = seeds[run_i]
        with timeit('%s %d %d' % (mth, run_i, seed)):
            # Evaluate
            config_list, perf_list, time_list = evaluate(mth, run_i, seed)

            # Save result
            print('=' * 20)
            print(seed, mth, config_list, perf_list, time_list)
            print(seed, mth, 'best perf', np.min(perf_list))

            timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            dir_path = 'logs/benchmark_surrogate/%s_%d/%s/' % (problem_str, max_runs, mth)
            file = 'benchmark_%s_%04d_%s.pkl' % (mth, seed, timestamp)
            try:
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
            except FileExistsError:
                pass
            with open(os.path.join(dir_path, file), 'wb') as f:
                save_item = (config_list, perf_list, time_list)
                pkl.dump(save_item, f)
            print(dir_path, file, 'saved!', flush=True)
