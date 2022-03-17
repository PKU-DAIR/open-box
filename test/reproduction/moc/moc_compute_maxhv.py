"""
example cmdline:

python test/reproduction/moc/moc_compute_maxhv.py --problem srn

"""
import os
import sys
import time
import numpy as np
import argparse

sys.path.insert(0, os.getcwd())
from moc_benchmark_function import get_problem, plot_pf
from openbox.utils.multi_objective import Hypervolume
from test.reproduction.test_utils import timeit
from platypus import NSGAII, Problem, Real

parser = argparse.ArgumentParser()
parser.add_argument('--problem', type=str)

args = parser.parse_args()
problem_str = args.problem
problem = get_problem(problem_str)
print('===== problem :', problem_str)

dim = problem.dim
num_objs = problem.num_objs
num_constraints = problem.num_constraints
bounds = problem.bounds


def CMO(xi):
    xi = np.asarray(xi)
    res = problem.evaluate(xi)
    return res['objs'], res['constraints']


t0 = time.time()
nsgaii_problem = Problem(dim, num_objs, num_constraints)
for k in range(dim):
    nsgaii_problem.types[k] = Real(bounds[k][0], bounds[k][1])
nsgaii_problem.constraints[:] = "<=0"
nsgaii_problem.function = CMO
algorithm = NSGAII(nsgaii_problem, population_size=1000)
algorithm.run(20000)

cheap_pareto_front = np.array([list(solution.objectives) for solution in algorithm.result])
cheap_constraints_values = np.array([list(solution.constraints) for solution in algorithm.result])
print('pf shape =', cheap_pareto_front.shape, cheap_constraints_values.shape)

hv = Hypervolume(problem.ref_point).compute(cheap_pareto_front)
t1 = time.time()
print('ref point =', problem.ref_point)
print('nsgaii hv =', hv)
print('time =', t1 - t0)
plot_pf(problem, problem_str, 'nsgaii', cheap_pareto_front, None)

