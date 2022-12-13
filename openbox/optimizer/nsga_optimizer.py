# License: MIT

import time
import random
import copy
import numpy as np
from openbox import logger
from openbox.optimizer.nsga_base import NSGABase
from openbox.utils.constants import MAXINT
from openbox.utils.platypus_utils import get_variator, set_problem_types, objective_wrapper
from openbox.utils.config_space import Configuration
from openbox.utils.util_funcs import deprecate_kwarg
from platypus import Problem, NSGAII
from platypus import nondominated as _nondominated


class NSGAOptimizer(NSGABase):
    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    def __init__(
            self,
            objective_function: callable,
            config_space,
            num_objectives=1,
            num_constraints=0,
            max_runs=2500,
            algorithm='nsgaii',
            logging_dir='logs',
            task_id='OpenBox',
            random_state=None,
            logger_kwargs: dict = None,
            **kwargs,
    ):

        if task_id is None:
            raise ValueError('Task id is not SPECIFIED. Please input task id first.')

        self.num_inputs = len(config_space.get_hyperparameters())
        self.num_constraints = num_constraints
        self.num_objectives = num_objectives
        self.algo = algorithm
        self.FAILED_PERF = [np.inf] * num_objectives
        super().__init__(objective_function, config_space, task_id=task_id, output_dir=logging_dir,
                         random_state=random_state, max_runs=max_runs, logger_kwargs=logger_kwargs)
        random.seed(self.rng.randint(MAXINT))

        # prepare objective function for platypus algorithm
        self.nsga_objective = objective_wrapper(objective_function, config_space, num_constraints)

        # set problem
        self.problem = Problem(self.num_inputs, num_objectives, num_constraints)
        set_problem_types(config_space, self.problem)
        if num_constraints > 0:
            self.problem.constraints[:] = "<=0"
        self.problem.function = self.nsga_objective

        # set algorithm
        if self.algo == 'nsgaii':
            population_size = kwargs.get('population_size', 100)
            if self.max_iterations <= population_size:
                logger.warning('max_runs <= population_size! Please check.')
                population_size = min(max_runs, population_size)
            variator = get_variator(config_space)
            self.algorithm = NSGAII(self.problem, population_size=population_size, variator=variator)
        else:
            raise ValueError('Unsupported algorithm: %s' % self.algo)

    def run(self):
        logger.info('Start optimization. max_iterations: %d' % (self.max_iterations,))
        start_time = time.time()
        self.algorithm.run(self.max_iterations)
        end_time = time.time()
        logger.info('Optimization is complete. Time: %.2fs.' % (end_time - start_time))
        return self

    def get_incumbents(self):
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
