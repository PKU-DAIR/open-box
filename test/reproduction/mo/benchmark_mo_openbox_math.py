"""
example cmdline:

python test/reproduction/mo/benchmark_mo_openbox_math.py --problem zdt2-3 --n 200 --init_strategy sobol --rep 1 --start_id 0

"""
import os
NUM_THREADS = "2"
os.environ["OMP_NUM_THREADS"] = NUM_THREADS         # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS    # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = NUM_THREADS         # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS     # export NUMEXPR_NUM_THREADS=1

import sys
import time
import numpy as np
import argparse
import pickle as pkl

sys.path.insert(0, os.getcwd())
from mo_benchmark_function import get_problem, plot_pf
from openbox import Optimizer
from openbox.utils.multi_objective import Hypervolume
from test.reproduction.test_utils import timeit, seeds

parser = argparse.ArgumentParser()
parser.add_argument('--problem', type=str)
parser.add_argument('--n', type=int, default=100)
parser.add_argument('--init', type=int, default=0)
parser.add_argument('--init_strategy', type=str, default='sobol', choices=['sobol', 'latin_hypercube'])
parser.add_argument('--surrogate', type=str, default='auto', choices=['auto', 'gp', 'prf'])
parser.add_argument('--acq_type', type=str, default='auto', choices=['auto', 'ehvi', 'usemo', 'mesmo'])
parser.add_argument('--optimizer', type=str, default='auto', choices=['auto', 'scipy', 'local'])
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--plot_mode', type=int, default=0)

args = parser.parse_args()
problem_str = args.problem
max_runs = args.n
initial_runs = args.init
init_strategy = args.init_strategy
surrogate_type = args.surrogate
acq_type = args.acq_type
if args.optimizer == 'scipy':
    acq_optimizer_type = 'random_scipy'
elif args.optimizer == 'local':
    acq_optimizer_type = 'local_random'
elif args.optimizer == 'auto':
    acq_optimizer_type = 'auto'
else:
    raise ValueError('Unknown optimizer %s' % args.optimizer)
if acq_type in ['usemo', 'mesmo']:
    acq_optimizer_type = None
rep = args.rep
start_id = args.start_id
plot_mode = args.plot_mode
if acq_type == 'ehvi':
    mth = 'openbox'
else:
    mth = 'openbox-%s' % acq_type

problem = get_problem(problem_str)
if initial_runs == 0:
    initial_runs = 2 * (problem.dim + 1)
cs = problem.get_configspace(optimizer='smac')
# max_runtime_per_trial = 600
task_id = '%s_%s_%s' % (mth, acq_type, problem_str)


def evaluate(mth, run_i, seed):
    print(mth, run_i, seed, '===== start =====', flush=True)

    def objective_function(config):
        res = problem.evaluate_config(config)
        # res['config'] = config
        return res

    bo = Optimizer(
        objective_function,
        cs,
        num_objectives=problem.num_objectives,
        num_constraints=0,
        surrogate_type=surrogate_type,            # default: auto: gp
        acq_type=acq_type,                        # default: auto: ehvi
        acq_optimizer_type=acq_optimizer_type,    # default: auto: random_scipy
        initial_runs=initial_runs,                # default: 2 * (problem.dim + 1)
        init_strategy=init_strategy,              # default: sobol
        max_runs=max_runs,
        ref_point=problem.ref_point, task_id=task_id, random_state=seed,
    )

    # bo.run()
    hv_diffs = []
    time_list = []
    global_start_time = time.time()
    for i in range(max_runs):
        observation = bo.iterate(bo.time_left)
        config, trial_state, objectives = observation.config, observation.trial_state, observation.objectives
        global_time = time.time() - global_start_time
        bo.time_left -= global_time
        print(seed, i, objectives, config, trial_state, 'time=', global_time)
        hv = Hypervolume(problem.ref_point).compute(bo.get_history().get_pareto_front())
        hv_diff = problem.max_hv - hv
        print(seed, i, 'hypervolume =', hv)
        print(seed, i, 'hv diff =', hv_diff)
        hv_diffs.append(hv_diff)
        time_list.append(global_time)
    config_list = bo.get_history().configurations
    perf_list = bo.get_history().get_objectives(transform='none')
    pf = np.asarray(bo.get_history().get_pareto_front())

    # plot for debugging
    if plot_mode == 1:
        Y_init = None
        plot_pf(problem, problem_str, mth, pf, Y_init)

    history = bo.get_history()

    return hv_diffs, pf, config_list, perf_list, time_list, history

if __name__ == '__main__':
    with timeit('%s all' % (mth,)):
        for run_i in range(start_id, start_id + rep):
            seed = seeds[run_i]
            with timeit('%s %d %d' % (mth, run_i, seed)):
                # Evaluate
                hv_diffs, pf, config_list, perf_list, time_list, history = evaluate(mth, run_i, seed)

                # Save result
                print('=' * 20)
                print(seed, mth, config_list, perf_list, time_list, hv_diffs)
                print(seed, mth, 'best hv_diff:', hv_diffs[-1])
                print(seed, mth, 'max_hv:', problem.max_hv)
                if pf is not None:
                    print(seed, mth, 'pareto num:', pf.shape[0])

                timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
                dir_path = 'logs/mo_benchmark_%s_%d/%s/' % (problem_str, max_runs, mth)
                file = 'benchmark_%s_%04d_%s.pkl' % (mth, seed, timestamp)
                os.makedirs(dir_path, exist_ok=True)
                with open(os.path.join(dir_path, file), 'wb') as f:
                    save_item = (hv_diffs, pf, config_list, perf_list, time_list)
                    pkl.dump(save_item, f)
                print(dir_path, file, 'saved!', flush=True)

                history.save_json(os.path.join(dir_path, 'benchmark_%s_%04d_%s.json' % (mth, seed, timestamp)))
