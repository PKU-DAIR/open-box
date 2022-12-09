from typing import Optional, Callable, List

import numpy as np
from ConfigSpace import ConfigurationSpace, Configuration, CategoricalHyperparameter
from ConfigSpace.hyperparameters import NumericalHyperparameter, OrdinalHyperparameter

from openbox import logger
from openbox.core.ea.base_ea_advisor import Individual, pareto_sort
from openbox.core.ea.base_modular_ea_advisor import ModularEAAdvisor
from openbox.utils.util_funcs import deprecate_kwarg


class CMAESEAAdvisor(ModularEAAdvisor):

    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    def __init__(self, config_space: ConfigurationSpace,
                 num_objectives=1,
                 num_constraints=0,
                 population_size=None,
                 optimization_strategy='ea',
                 constraint_strategy='discard',
                 batch_size=1,
                 output_dir='logs',
                 task_id='OpenBox',
                 random_state=None,

                 required_evaluation_count: Optional[int] = None,
                 auto_step=True,
                 strict_auto_step=True,
                 skip_gen_population=False,
                 filter_gen_population: Optional[Callable[[List[Configuration]], List[Configuration]]] = None,
                 keep_unexpected_population=True,
                 save_cached_configuration=True,

                 mu=None,
                 w=None,
                 cs=None,
                 ds=None,
                 cc=None,
                 mu_cov=None,
                 c_cov=None,
                 random_starting_mean=False):

        self.n = len(config_space.keys())
        required_evaluation_count = required_evaluation_count or population_size

        if population_size is None:
            population_size = (4 + int(3 * np.log(self.n))) * 3

        super().__init__(config_space=config_space, num_objectives=num_objectives, num_constraints=num_constraints,
                         population_size=population_size, optimization_strategy=optimization_strategy,
                         batch_size=batch_size, output_dir=output_dir, task_id=task_id,
                         random_state=random_state,

                         required_evaluation_count=required_evaluation_count, auto_step=auto_step,
                         strict_auto_step=strict_auto_step, skip_gen_population=skip_gen_population,
                         filter_gen_population=filter_gen_population,
                         keep_unexpected_population=keep_unexpected_population,
                         save_cached_configuration=save_cached_configuration
                         )

        self.constraint_strategy = constraint_strategy
        assert self.constraint_strategy in {'discard', 'dominated'}

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

        self.generation_id = 0

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

    def _gen(self, count=1) -> List[Configuration]:
        array = np.random.multivariate_normal(self.mean, self.cov * (self.sigma * self.sigma))
        while array.max() > 1 or array.min() < 0:
            array = np.random.multivariate_normal(self.mean, self.cov * (self.sigma * self.sigma))

        array1 = self.validate_array(array)
        array2 = self.unnormalize(array1)

        config = Configuration(self.config_space, vector=array2)
        self.unvalidated_map[config] = array
        return [config]

    def _could_sel(self) -> bool:
        if self.constraint_strategy == 'discard' and self.num_constraints > 0:
            self.next_population = [x for x in self.next_population if x.constraints_satisfied]
        return super()._could_sel()

    def _sel(self, parent: List[Individual], sub: List[Individual]) -> List[Individual]:
        mean1 = np.zeros_like(self.mean)
        sub = pareto_sort(sub)

        if self.constraint_strategy == 'discard' and self.num_constraints > 0:
            self.next_population = [x for x in self.next_population if x.constraints_satisfied]
        if self.constraint_strategy == 'dominated' and self.num_constraints > 0:
            good = [x for x in sub if x.constraints_satisfied]
            bad = [x for x in sub if not x.constraints_satisfied]
            sub = good + bad

        pop_arrays = [self.unvalidated_map[x.config] for x in sub]

        for i in range(self.mu):
            mean1 += self.w[i] * pop_arrays[i]

        si_cov = self.mat_sqrt_inv(self.cov)

        meand = mean1 - self.mean

        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * (
                si_cov @ meand) / self.sigma

        nps = np.linalg.norm(self.ps)
        e_n_0i = np.sqrt(self.n) * (1 - 1 / (4 * self.n) + 1 / (21 * self.n * self.n))

        sigma1 = self.sigma * np.exp(self.cs / self.ds * (nps / e_n_0i - 1))

        hs = 1 if nps / np.sqrt(1 - (1 - self.cs) ** (2 * (self.generation_id + 1))) < (
                1.5 + 1 / (self.n - 0.5)) * e_n_0i else 0

        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) / self.sigma * meand

        dh = (1 - hs) * self.cc * (2 - self.cc)

        self.cov = (1 - self.c_cov) * self.cov + (self.c_cov / self.mu_cov) * (
                    self.op(self.pc) + dh * self.cov) + self.c_cov * (1 - 1 / self.mu_cov) * np.array(
            [self.w[i] * self.op((pop_arrays[i] - self.mean) / self.sigma) for i in range(self.mu)]).sum(axis=0)

        self.mean = mean1
        self.sigma = sigma1

        if np.linalg.det(self.cov) == 0.0:
            noise = np.random.random(self.cov.shape)
            noise = noise + noise.T
            self.cov += noise * np.average(self.cov) * 0.00001
            logger.warning("Covariance matrix not full rank! Adding a noise to it.")

        self.generation_id += 1
        self.population = list(sub)

        self.unvalidated_map = dict()

        return list(sub)
