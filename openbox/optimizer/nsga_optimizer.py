# License: MIT

import sys
import time
import traceback
import random
import copy
import numpy as np
from openbox.optimizer.nsga_base import NSGABase
from openbox.utils.constants import MAXINT
from openbox.utils.platypus_utils import get_variator, set_problem_types, objective_wrapper
from openbox.utils.config_space import Configuration
from platypus import Problem, NSGAII
from platypus import nondominated as _nondominated

"""
    The objective function returns a dictionary that has --- config, constraints, objs ---.
"""


class NSGAOptimizer(NSGABase):
    def __init__(self, objective_function: callable,
                 config_space,
                 num_constraints=0,
                 num_objs=1,
                 max_runs=2500,
                 algorithm='nsgaii',
                 logging_dir='logs',
                 task_id='default_task_id',
                 random_state=None,
                 **kwargs):

        if task_id is None:
            raise ValueError('Task id is not SPECIFIED. Please input task id first.')

        self.num_inputs = len(config_space.get_hyperparameters())
        self.num_constraints = num_constraints
        self.num_objs = num_objs
        self.algo = algorithm
        self.FAILED_PERF = [MAXINT] * num_objs
        super().__init__(objective_function, config_space, task_id=task_id, output_dir=logging_dir,
                         random_state=random_state, max_runs=max_runs)
        random.seed(self.rng.randint(MAXINT))

        # prepare objective function for platypus algorithm
        self.nsga_objective = objective_wrapper(objective_function, config_space, num_constraints)

        # set problem
        self.problem = Problem(self.num_inputs, num_objs, num_constraints)
        set_problem_types(config_space, self.problem)
        if num_constraints > 0:
            self.problem.constraints[:] = "<=0"
        self.problem.function = self.nsga_objective

        # set algorithm
        if self.algo == 'nsgaii':
            population_size = kwargs.get('population_size', 100)
            if self.max_iterations <= population_size:
                self.logger.warning('max_runs <= population_size! Please check.')
                population_size = min(max_runs, population_size)
            variator = get_variator(config_space)
            self.algorithm = NSGAII(self.problem, population_size=population_size, variator=variator)
        else:
            raise ValueError('Unsupported algorithm: %s' % self.algo)

    def run(self):
        self.logger.info('Start optimization. max_iterations: %d' % (self.max_iterations,))
        start_time = time.time()
        self.algorithm.run(self.max_iterations)
        end_time = time.time()
        self.logger.info('Optimization is complete. Time: %.2fs.' % (end_time - start_time))
        return self

    def get_incumbent(self):
        solutions = self.get_solutions(feasible=True, nondominated=True, decode=True)
        pareto_set = [Configuration(self.config_space, vector=np.asarray(s.variables)) for s in solutions]
        pareto_front = np.array([s.objectives for s in solutions])
        return pareto_set, pareto_front

    def get_pareto_set(self):
        solutions = self.get_solutions(feasible=True, nondominated=True, decode=True)
        pareto_set = [Configuration(self.config_space, vector=np.asarray(s.variables)) for s in solutions]
        return pareto_set

    def get_pareto_front(self):
        solutions = self.get_solutions(feasible=True, nondominated=True, decode=True)
        pareto_front = np.array([s.objectives for s in solutions])
        return pareto_front

    def get_solutions(self, feasible=True, nondominated=True, decode=True):
        solutions = copy.deepcopy(self.algorithm.result)
        if feasible:
            solutions = [s for s in solutions if s.feasible]
        if nondominated:
            solutions = _nondominated(solutions)
        if decode:
            for s in solutions:
                s.variables[:] = [self.problem.types[i].decode(s.variables[i]) for i in range(self.problem.nvars)]
        return solutions
