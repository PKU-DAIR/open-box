"""
example cmdline:

python test/optimizer/so/benchmark_so_gpflowopt_math.py --problem branin --n 200 --init 3 --rep 1 --start_id 0

"""
import os
import sys
import time
import numpy as np
import argparse
import pickle as pkl

import gpflow
# from gpflowopt.bo import BayesianOptimizer
from gpflowopt.design import LatinHyperCube,RandomDesign
from gpflowopt.acquisition import ExpectedImprovement, ProbabilityOfFeasibility
from gpflowopt.optim import SciPyOptimizer, StagedOptimizer, MCOptimizer

sys.path.insert(0, os.getcwd())
from test.reproduction.soc.soc_benchmark_function import get_problem
from openbox.test_utils import timeit, seeds
from test.reproduction.gpflowopt_modified import BayesianOptimizer_modified

parser = argparse.ArgumentParser()
parser.add_argument('--problem', type=str, default='townsend')
parser.add_argument('--nc', type=int, default=1)
parser.add_argument('--n', type=int, default=200)
parser.add_argument('--init', type=int, default=3)
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)

args = parser.parse_args()
problem_str = args.problem
max_runs = args.n
num_constraints = args.nc
initial_runs = args.init
rep = args.rep
start_id = args.start_id
mth = 'gpflowopt'

problem = get_problem(problem_str)
domain = problem.get_configspace(optimizer='gpflowopt')


def evaluate(mth, run_i, seed):
    print(mth, run_i, seed, '===== start =====', flush=True)

    def objective_function(x):
        y = problem.evaluate(x)
        return np.array([[y['objs'][0]]])

    def constraint_function(x):
        y = problem.evaluate(x)
        return np.array([[y['constraints'][0]]])

    # random seed
    np.random.seed(seed)

    # Initial evaluations
    X_init = LatinHyperCube(initial_runs, domain).generate()
    # X_init = RandomDesign(initial_runs, domain).generate()
    Y_init = np.vstack([objective_function(X_init[i, :]) for i in range(X_init.shape[0])])
    C_init = np.vstack([constraint_function(X_init[i, :]) for i in range(X_init.shape[0])])

    # Use standard Gaussian process Regression
    model = gpflow.gpr.GPR(X_init, Y_init, gpflow.kernels.Matern52(domain.size, ARD=True))
    model.kern.lengthscales.transform = gpflow.transforms.Log1pe(1e-3)

    constraint_model = gpflow.gpr.GPR(X_init, C_init, gpflow.kernels.Matern52(domain.size, ARD=True))
    constraint_model.kern.lengthscales.transform = gpflow.transforms.Log1pe(1e-3)
    constraint_model.likelihood.variance = 0.01
    constraint_model.likelihood.variance.prior = gpflow.priors.Gamma(1. / 4., 1.0)

    # Now create the Bayesian Optimizer
    ei = ExpectedImprovement(model)
    pof = ProbabilityOfFeasibility(constraint_model)
    joint = ei * pof
    acquisition_opt = StagedOptimizer([MCOptimizer(domain, 400),
                                       SciPyOptimizer(domain)])
    optimizer = BayesianOptimizer_modified(domain, joint, optimizer=acquisition_opt, verbose=True)

    # Run the Bayesian optimization for (max_runs-init_num) iterations
    result = optimizer.optimize([objective_function, constraint_function], n_iter=max_runs)

    # Save result
    X, Y = optimizer.acquisition.data

    perf_list = [_[0] for _ in Y]
    time_list = [0.] * initial_runs + optimizer.time_list

    return X, perf_list, time_list


with timeit('%s all' % (mth,)):
    for run_i in range(start_id, start_id + rep):
        seed = seeds[run_i]
        np.random.seed(seed)
        with timeit('%s %d %d' % (mth, run_i, seed)):
            # Evaluate
            X, perf_list, time_list = evaluate(mth, run_i, seed)

            # Save result
            print('=' * 20)
            print(seed, mth, X, perf_list, time_list)
            print(seed, mth, 'best perf', np.min(perf_list))

            timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            dir_path = 'logs/soc_benchmark_%s_%d/%s/' % (problem_str, max_runs, mth)
            file = 'benchmark_%s_%04d_%s.pkl' % (mth, seed, timestamp)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            with open(os.path.join(dir_path, file), 'wb') as f:
                save_item = (X, perf_list, time_list)
                pkl.dump(save_item, f)
            print(dir_path, file, 'saved!', flush=True)
