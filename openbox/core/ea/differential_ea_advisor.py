import random
from typing import Optional, Callable, List, Union, Tuple

from ConfigSpace import Configuration, ConfigurationSpace, CategoricalHyperparameter, OrdinalHyperparameter
from ConfigSpace.hyperparameters import NumericalHyperparameter

from openbox.core.ea.base_ea_advisor import Individual, pareto_best, pareto_sort
from openbox.core.ea.base_modular_ea_advisor import ModularEAAdvisor
from openbox.utils.util_funcs import deprecate_kwarg


class DifferentialEAAdvisor(ModularEAAdvisor):

    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    def __init__(self,

                 config_space: ConfigurationSpace,
                 num_objectives=1,
                 num_constraints=0,
                 population_size=30,
                 optimization_strategy='ea',
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

                 constraint_strategy='discard',

                 f: Union[Tuple[float, float], float] = 0.5,
                 cr: Union[Tuple[float, float], float] = 0.9,
                 ):
        """
        f is the hyperparameter for DEA that X = A + (B - C) * f
        cr is the cross rate
        f and cr may be a tuple of two floats, such as (0.1,0.9)
        If so, these two values are adjusted automatically within this range.
        """
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
        assert self.constraint_strategy in {'discard'}

        self.f = f
        self.cr = cr

        self.dynamic_f = isinstance(f, tuple)
        self.dynamic_cr = isinstance(cr, tuple)

        assert self.num_objectives == 1 or not (self.dynamic_f or self.dynamic_cr)

        self.iter = None
        self.cur = 0

        self.nid_map = {}

    def _gen(self, count=1) -> List[Configuration]:
        if len(self.population) < self.population_size:
            next_config = self.sample_random_config(excluded_configs=self.all_configs)
            nid = -1
        else:
            xi = self.population[self.cur]['config']
            xi_score = self.population[self.cur]['perf']

            # Randomly sample 3 other values: x1, x2, x3
            lst = list(range(self.population_size))
            lst.remove(self.cur)
            random.shuffle(lst)
            lst = lst[:3]

            if self.dynamic_f:
                lst.sort(key=lambda a: self.population[a]['perf'])

            i1, i2, i3 = lst[0], lst[1], lst[2]
            x1, x2, x3 = self.population[i1]['config'], self.population[i2]['config'], self.population[i3]['config']

            # Mutation: xt = x1 + (x2 - x3) * f
            if self.dynamic_f:
                # Dynamic f
                f1, f2, f3 = self.population[i1]['perf'], self.population[i2]['perf'], self.population[i3]['perf']
                if f1 == f3:
                    f = self.f[0]
                else:
                    f = self.f[0] + (self.f[1] - self.f[0]) * (f1 - f2) / (f1 - f3)
            else:
                # Fixed f
                f = self.f

            xt = self.mutate(x1, x2, x3, f)

            # Cross over between xi and xt, get xn
            if self.dynamic_cr:
                # Dynamic cr
                scores = [a['perf'] for a in self.population]
                scores_avg = sum(scores) / len(scores)

                if xi_score < scores_avg:
                    scores_mx = max(scores)
                    scores_mn = min(scores)
                    cr = self.cr[0] + (self.cr[1] - self.cr[0]) * (scores_mx - xi_score) / max(
                        scores_mx - scores_mn, 1e-10)
                else:
                    cr = self.cr[0]
            else:
                # Fixed cr
                cr = self.cr

            xn = self.cross_over(xi, xt, cr)

            # xn should be evaluated.
            # if xn is better than xi, we replace xi with xn.

            # xn = get_one_exchange_neighbourhood(xi)

            next_config = xn
            nid = self.cur
            self.cur = (self.cur + 1) % self.population_size

        self.nid_map[next_config] = nid
        return [next_config]

    def _sel(self, parent: List[Individual], sub: List[Individual]) -> List[Individual]:
        if self.constraint_strategy == 'discard' and self.num_constraints > 0:
            sub = [x for x in sub if x.constraints_satisfied]

        for conf in sub:
            if conf in self.nid_map and self.nid_map[conf] != -1:
                i = self.nid_map[conf]
                if i < len(parent):
                    conf0 = parent[i]
                    if pareto_best([conf, conf0], 1) == conf:
                        parent[i] = conf
                else:
                    parent.append(conf)

        for conf in sub:
            if conf not in self.nid_map or self.nid_map[conf] == -1:
                parent.append(conf)

        parent = pareto_sort(parent)
        parent = parent[:self.population_size]
        random.shuffle(parent)

        return parent

    def mutate(self, config_a: Configuration, config_b: Configuration, config_c: Configuration, f: float):
        """
        Compute A + (B - C) * f. Basically element-wise.
        For ranged int/float values, the result will be clamped into [lower, upper].
        For categorical/ordinal values, the values are converted to ints and the result is (mod SIZE).
        e. g. in ["A", "B", "C", "D"], "D" + "B" - "A" => 3 + 1 - 0 => 4 => 0 (mod 4) => "A"
        """
        new_array = config_a.get_array() + (config_b.get_array() - config_c.get_array()) * f

        for i, key in enumerate(self.config_space.keys()):
            hp_type = self.config_space.get_hyperparameter(key)
            if isinstance(hp_type, CategoricalHyperparameter) or isinstance(hp_type, OrdinalHyperparameter):
                v = (round(new_array[i]) % hp_type.get_size() + hp_type.get_size()) % hp_type.get_size()
                new_array[i] = v
            elif isinstance(hp_type, NumericalHyperparameter):
                # new_array[i] = max(0, min(new_array[i], 1))
                if new_array[i] < 0:
                    # new_array[i] = -new_array[i] / 2
                    new_array[i] = random.random()
                if new_array[i] > 1:
                    # new_array[i] = 1 - (new_array[i] - 1) / 2
                    new_array[i] = random.random()
            else:
                pass

        config = Configuration(self.config_space, vector=new_array)
        return config

    def cross_over(self, config_a: Configuration, config_b: Configuration, cr: float):
        """
        The cross-over operation.
        For each element of config_a, it has cr possibility to be replaced with that of config_b.
        """
        a1, a2 = config_a.get_array(), config_b.get_array()
        any_changed = False

        for i in range(len(self.config_space.keys())):
            if self.rng.random() < cr:
                a1[i] = a2[i]  # a1, a2 are vector copies, modification is ok.
                any_changed = True

        # Make sure cross-over changes at least one dimension. Otherwise it makes no sense.
        if not any_changed:
            i = self.rng.randint(0, len(self.config_space.keys()) - 1)
            a1[i] = a2[i]

        return Configuration(self.config_space, vector=a1)
