# License: MIT
import random
from typing import Optional, Callable, List

from ConfigSpace import Configuration

from openbox.core.ea.base_modular_ea_advisor import ModularEAAdvisor
from openbox.utils.constants import MAXINT
from openbox.utils.config_space import get_one_exchange_neighbourhood
from openbox.core.ea.base_ea_advisor import Individual, pareto_sort
from openbox.utils.util_funcs import deprecate_kwarg


class RegularizedEAAdvisor(ModularEAAdvisor):
    """
    Evolutionary Algorithm Advisor
    """

    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    def __init__(self,

                 config_space,
                 num_objectives=1,
                 num_constraints=0,
                 population_size=30,
                 optimization_strategy='ea',
                 batch_size=1,
                 output_dir='logs',
                 task_id='OpenBox',
                 random_state=None,

                 constraint_strategy='discard',

                 required_evaluation_count: Optional[int] = 1,
                 auto_step=True,
                 strict_auto_step=True,
                 skip_gen_population=False,
                 filter_gen_population: Optional[Callable[[List[Configuration]], List[Configuration]]] = None,
                 keep_unexpected_population=True,
                 save_cached_configuration=True,

                 subset_size=20,
                 epsilon=0.2,
                 strategy='worst',
                 ):

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

        # assert num_objectives == 1
        assert constraint_strategy == 'discard'
        self.constraint_strategy = constraint_strategy

        self.subset_size = subset_size
        assert 0 < self.subset_size <= self.population_size
        self.epsilon = epsilon
        self.strategy = strategy
        assert self.strategy in ['worst', 'oldest']

    def _gen(self, count=1) -> List[Configuration]:

        if len(self.population) < self.population_size:
            # Initialize population
            next_config = self.sample_random_config(excluded_configs=self.all_configs)
        else:
            # Select a parent by subset tournament and epsilon greedy
            if self.rng.random() < self.epsilon:
                parent_config = random.sample(self.population, 1)[0].config
            else:
                subset = random.sample(self.population, self.subset_size)
                subset = pareto_sort(subset)
                parent_config = subset[0].config

            # Mutation to 1-step neighbors
            next_config = None
            neighbors_gen = get_one_exchange_neighbourhood(parent_config, seed=self.rng.randint(MAXINT))
            for neighbor in neighbors_gen:
                if neighbor not in self.all_configs:
                    next_config = neighbor
                    break
            if next_config is None:  # If all the neighors are evaluated, sample randomly!
                next_config = self.sample_random_config(excluded_configs=self.all_configs)

        self.all_configs.add(next_config)

        return [next_config]

    def _sel(self, parent: List[Individual], sub: List[Individual]) -> List[Individual]:
        if self.constraint_strategy == 'discard' and self.num_constraints > 0:
            sub = [o for o in sub if o.constraints_satisfied]

        for i in sub:
            i.data['age'] = self.age
            self.age += 1

        parent.extend(sub)

        if len(parent) > self.population_size:
            if self.strategy == 'oldest':
                parent.sort(key=lambda x: x['age'], reverse=True)
                parent = parent[:self.population_size]
            elif self.strategy == 'worst':
                parent = pareto_sort(parent)
                parent = parent[:self.population_size]
            else:
                raise ValueError('Unknown strategy: %s' % self.strategy)

        return parent
