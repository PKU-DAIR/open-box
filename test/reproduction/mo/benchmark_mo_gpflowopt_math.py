"""
example cmdline:

python test/reproduction/mo/benchmark_mo_gpflowopt_math.py --problem dtlz2-12-2 --n 200 --init 10 --mc 1000 --rep 1 --start_id 0

"""
import os
import sys
import time
import numpy as np
import argparse
import pickle as pkl

import gpflow
import gpflowopt

sys.path.insert(0, os.getcwd())
from mo_benchmark_function import get_problem, plot_pf
from openbox.utils.multi_objective import Hypervolume
from test.reproduction.test_utils import timeit, seeds
from test.reproduction.gpflowopt_modified import BayesianOptimizer_modified

parser = argparse.ArgumentParser()
parser.add_argument('--problem', type=str)
parser.add_argument('--n', type=int, default=100)
parser.add_argument('--init', type=int, default=10)
parser.add_argument('--mc', type=int, default=1000)
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--plot_mode', type=int, default=0)

args = parser.parse_args()
problem_str = args.problem
max_runs = args.n
initial_runs = args.init
optimizer_mc_times = args.mc
rep = args.rep
start_id = args.start_id
plot_mode = args.plot_mode
mth = 'gpflowopt'

problem = get_problem(problem_str)
domain = problem.get_configspace(optimizer='gpflowopt')


def evaluate(mth, run_i, seed):
    print(mth, run_i, seed, '===== start =====', flush=True)

    def objective_function(x):
        res = problem.evaluate(x)
        return np.array(res['objs']).reshape(1, -1)

    # random seed
    np.random.seed(seed)

    # Initial evaluations
    X_init = gpflowopt.design.LatinHyperCube(initial_runs, domain).generate()
    # X_init = gpflowopt.design.RandomDesign(initial_runs, domain).generate()
    # fix numeric problem
    if hasattr(problem, 'lb') and hasattr(problem, 'ub'):
        eps = 1e-8
        X_init = np.maximum(X_init, problem.lb + eps)
        X_init = np.minimum(X_init, problem.ub - eps)
    Y_init = np.vstack([objective_function(X_init[i, :]) for i in range(X_init.shape[0])])

    # One model for each objective
    objective_models = [gpflow.gpr.GPR(X_init.copy(), Y_init[:, [i]].copy(),
                                       gpflow.kernels.Matern52(domain.size, ARD=True))
                        for i in range(Y_init.shape[1])]
    for model in objective_models:
        model.likelihood.variance = 0.01

    hvpoi = gpflowopt.acquisition.HVProbabilityOfImprovement(objective_models)
    # First setup the optimization strategy for the acquisition function
    # Combining MC step followed by L-BFGS-B
    acquisition_opt = gpflowopt.optim.StagedOptimizer([gpflowopt.optim.MCOptimizer(domain, optimizer_mc_times),
                                                       gpflowopt.optim.SciPyOptimizer(domain)])

    # Then run the BayesianOptimizer for (max_runs-init_num) iterations
    optimizer = BayesianOptimizer_modified(domain, hvpoi, optimizer=acquisition_opt, verbose=True)
    result = optimizer.optimize(objective_function, n_iter=max_runs-initial_runs)

    # Save result
    # pf = optimizer.acquisition.pareto.front.value
    # pf, dom = gpflowopt.pareto.non_dominated_sort(hvpoi.data[1])
    pf = gpflowopt.pareto.Pareto(optimizer.acquisition.data[1]).front.value
    X, Y = optimizer.acquisition.data
    time_list = [0.] * initial_runs + optimizer.time_list
    hv_diffs = []
    for i in range(Y.shape[0]):
        # hv = gpflowopt.pareto.Pareto(Y[:i+1]).hypervolume(problem.ref_point)    # ref_point problem
        hv = Hypervolume(problem.ref_point).compute(Y[:i+1])
        hv_diff = problem.max_hv - hv
        hv_diffs.append(hv_diff)

    # plot for debugging
    if plot_mode == 1:
        plot_pf(problem, problem_str, mth, pf, Y_init)

    return hv_diffs, pf, X, Y, time_list


with timeit('%s all' % (mth,)):
    for run_i in range(start_id, start_id + rep):
        seed = seeds[run_i]
        with timeit('%s %d %d' % (mth, run_i, seed)):
            # Evaluate
            hv_diffs, pf, X, Y, time_list = evaluate(mth, run_i, seed)

            # Save result
            print('=' * 20)
            print(seed, mth, X, Y, time_list, hv_diffs)
            print(seed, mth, 'best hv_diff:', hv_diffs[-1])
            print(seed, mth, 'max_hv:', problem.max_hv)
            if pf is not None:
                print(seed, mth, 'pareto num:', pf.shape[0])

            timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            dir_path = 'logs/mo_benchmark_%s_%d/%s/' % (problem_str, max_runs, mth)
            file = 'benchmark_%s_%04d_%s.pkl' % (mth, seed, timestamp)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            with open(os.path.join(dir_path, file), 'wb') as f:
                save_item = (hv_diffs, pf, X, Y, time_list)
                pkl.dump(save_item, f)
            print(dir_path, file, 'saved!', flush=True)
