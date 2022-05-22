import random

from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, OrdinalHyperparameter

from typing import *

from ConfigSpace.hyperparameters import NumericalHyperparameter

from openbox import Observation

from openbox.core.ea.base_ea_advisor import *
from openbox.surrogate.base.base_model import AbstractModel
from openbox.utils.constants import MAXINT, SUCCESS


class CMAESEAAdvisor(EAAdvisor):

    def __init__(self, config_space: ConfigurationSpace,
                 num_objs=1,
                 num_constraints=0,
                 population_size=None,
                 optimization_strategy='ea',
                 constraint_strategy='discard',
                 batch_size=1,
                 output_dir='logs',
                 task_id='default_task_id',
                 random_state=None,

                 mu=None,
                 w=None,
                 cs=None,
                 ds=None,
                 cc=None,
                 mu_cov=None,
                 c_cov=None,
                 random_starting_mean=False):

        self.n = len(config_space.keys())

        if population_size is None:
            population_size = (4 + int(3 * np.log(self.n))) * 3

        EAAdvisor.__init__(self, config_space=config_space, num_objs=num_objs, num_constraints=num_constraints,
                           population_size=population_size, optimization_strategy=optimization_strategy,
                           batch_size=batch_size, output_dir=output_dir, task_id=task_id, random_state=random_state,
                           )

        self.constraint_strategy = constraint_strategy
        assert self.constraint_strategy in {'discard', 'dominated'}

        self.next_population: List[Individual] = []

        self.lam = self.population_size

        self.mu = mu if mu is not None else int(self.lam / 2)

        self.w = w if w is not None else \
            (lambda a: a / np.linalg.norm(a, ord=1))(
                np.array([np.log((self.mu + 1) / i) for i in range(1, self.mu + 1)]))

        self.mu_eff = 1 / (self.w ** 2).sum()

        self.cs = cs if cs is not None else (self.mu_eff + 2) / (self.n + self.mu_eff + 3)
        self.ds = ds if ds is not None else 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.n + 1)) - 1) + self.cs
        self.cc = cc if cc is not None else 4 / (self.n + 4)
        self.mu_cov = mu_cov if mu_cov is not None else self.mu_eff
        self.c_cov = c_cov if c_cov is not None else 2 / self.mu_cov / ((self.n + np.sqrt(2)) ** 2) + (
                1 - 1 / self.mu_cov) * min(1, (2 * self.mu_eff - 1) / ((self.n + 2) ** 2 + self.mu_eff))

        self.ps = np.zeros((self.n,))
        self.pc = np.zeros((self.n,))
        self.cov = np.eye(self.n)

        self.mean = np.random.random((self.n,)) if random_starting_mean else np.ones((self.n,)) / 2
        self.sigma = 0.5

        self.gen = 0

        self.unvalidated_map = dict()

    def validate_array(self, array):
        array1 = array.copy()
        for i, key in enumerate(self.config_space.keys()):
            array1[i] -= int(array1[i]) // 2 * 2
            if array1[i] < 0:
                array1[i] += 2
            if array1[i] > 1:
                array1[i] = 2 - array1[i]

        return array1

    def normalize(self, array):
        """
        normalize scales each dimension of the array into [0,1], for further convenience.
        """
        array1 = array.copy()
        for i, key in enumerate(self.config_space.keys()):
            hp_type = self.config_space.get_hyperparameter(key)
            if isinstance(hp_type, CategoricalHyperparameter) or isinstance(hp_type, OrdinalHyperparameter):
                array1[i] /= hp_type.get_size()
            elif isinstance(hp_type, NumericalHyperparameter):
                pass
            else:
                pass
        return array1

    def unnormalize(self, array):
        array1 = array.copy()
        for i, key in enumerate(self.config_space.keys()):
            hp_type = self.config_space.get_hyperparameter(key)
            if isinstance(hp_type, CategoricalHyperparameter) or isinstance(hp_type, OrdinalHyperparameter):
                array1[i] = int(array1[i] * hp_type.get_size())
            elif isinstance(hp_type, NumericalHyperparameter):
                pass
            else:
                pass
        return array1

    def mat_sqrt_inv(self, matrix):
        """
        For a positive-definite, symmetric matrix M. Calculate M ^ (-1/2).
        This is required in the algorithm.
        """
        e, v = np.linalg.eigh(matrix)
        e1 = e
        for i in range(e1.shape[0]):
            e1[i] = (e1[i] ** -0.5 if e1[i] > 0 else e1[i])
        ans = v @ np.diag(e1) @ v.T
        return ans

    def op(self, array):
        """
        Outer product
        """
        return array @ array.T

    def get_suggestion(self):

        if len(self.next_population) >= self.lam:
            mean1 = np.zeros_like(self.mean)
            self.next_population = pareto_sort(self.next_population)

            if self.constraint_strategy == 'dominated' and self.num_constraints > 0:
                good = [x for x in self.next_population if x.constraints_satisfied]
                bad = [x for x in self.next_population if not x.constraints_satisfied]
                self.next_population = good + bad

            pop_arrays = [self.unvalidated_map[x.config] for x in self.next_population]

            for i in range(self.mu):
                mean1 += self.w[i] * pop_arrays[i]

            si_cov = self.mat_sqrt_inv(self.cov)

            meand = mean1 - self.mean

            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * (
                    si_cov @ meand) / self.sigma

            nps = np.linalg.norm(self.ps)
            e_n_0i = np.sqrt(self.n) * (1 - 1 / (4 * self.n) + 1 / (21 * self.n * self.n))

            sigma1 = self.sigma * np.exp(self.cs / self.ds * (nps / e_n_0i - 1))

            hs = 1 if nps / np.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) < (
                    1.5 + 1 / (self.n - 0.5)) * e_n_0i else 0

            self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) / self.sigma * meand

            dh = (1 - hs) * self.cc * (2 - self.cc)

            self.cov = (1 - self.c_cov) * self.cov + (self.c_cov / self.mu_cov) * (
                    self.op(self.pc) + dh * self.cov) + self.c_cov * (1 - 1 / self.mu_cov) * np.array([
                self.w[i] * self.op((pop_arrays[i] - self.mean) / self.sigma) for i in
                range(self.mu)]).sum(axis=0)

            self.mean = mean1
            self.sigma = sigma1

            if np.linalg.det(self.cov) == 0.0:
                noise = np.random.random(self.cov.shape)
                noise = noise + noise.T
                self.cov += noise * np.average(self.cov) * 0.00001
                self.logger.warning("Covariance matrix not full rank! Adding a noise to it.")

            self.gen += 1
            self.population = list(self.next_population)
            self.next_population = []

            self.unvalidated_map = dict()

        array = np.random.multivariate_normal(self.mean, self.cov * (self.sigma * self.sigma))
        while True:
            if array.max() > 1 or array.min() < 0:
                # s *= 0.9
                array = np.random.multivariate_normal(self.mean, self.cov * (self.sigma * self.sigma))
                continue
            break

        array1 = self.validate_array(array)
        array2 = self.unnormalize(array1)

        config = Configuration(self.config_space, vector=array2)
        self.unvalidated_map[config] = array
        return config

    def update_observation(self, observation: Observation):
        config = observation.config
        constraint = constraint_check(observation.constraints)
        perf = observation.objs
        trial_state = observation.trial_state

        pop = Individual(config=config, perf=perf, constraints_satisfied=constraint)

        if trial_state == SUCCESS:

            if not constraint:
                if self.constraint_strategy == 'discard':
                    return self.history_container.update_observation(observation)

            self.next_population.append(pop)

        return self.history_container.update_observation(observation)
