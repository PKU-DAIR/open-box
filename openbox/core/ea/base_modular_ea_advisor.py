from typing import Optional, List, Callable

from ConfigSpace import ConfigurationSpace, Configuration

from openbox.core.ea.base_ea_advisor import EAAdvisor, Individual, as_individual
from openbox.utils.history import Observation
from openbox.utils.util_funcs import deprecate_kwarg


class ModularEAAdvisor(EAAdvisor):

    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    def __init__(self, config_space: ConfigurationSpace,
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
                 save_cached_configuration=True
                 ):

        super().__init__(config_space=config_space, num_objectives=num_objectives, num_constraints=num_constraints,
                         population_size=population_size, optimization_strategy=optimization_strategy,
                         batch_size=batch_size, output_dir=output_dir, task_id=task_id, random_state=random_state,
                         )

        self.required_evaluation_count = required_evaluation_count or self.population_size

        self.filled_up = False
        self.cached_config: List[Configuration] = list()
        self.uneval_config: List[Configuration] = list()
        self.next_population: List[Individual] = list()

        self.auto_step = auto_step
        self.strict_auto_step = strict_auto_step
        self.skip_gen_population = skip_gen_population
        self.filter_gen_population = filter_gen_population
        self.keep_unexpected_population = keep_unexpected_population
        self.save_cached_configuration = save_cached_configuration

    def generated_count(self) -> int:
        return len(self.cached_config) + len(self.uneval_config) + len(self.next_population)

    def _gen(self, count=1) -> List[Configuration]:
        raise NotImplementedError

    def _could_sel(self) -> bool:
        return len(self.next_population) >= self.required_evaluation_count

    def _sel(self, parent: List[Individual], sub: List[Individual]) -> List[Individual]:
        raise NotImplementedError

    def gen(self, count=None):

        if count is None:
            count = self.required_evaluation_count - len(self.cached_config)
        # if self.__class__.__name__ == 'RegularizedEAAdvisor':
        #             print('gen count is', count)

        temp = []

        while len(self.cached_config) + len(temp) < count:
            res = self._gen(count - len(self.cached_config) - len(temp))
            if not res:
                break
            temp.extend(res)

        if self.filter_gen_population is not None:
            temp = self.filter_gen_population(temp)

        self.cached_config.extend(temp)

    def could_sel(self):
        return self._could_sel()

    def sel(self):

        self.population = self._sel(self.population, self.next_population)
        self.filled_up = True

        self.next_population = []
        if not self.save_cached_configuration:
            self.cached_config.clear()
            self.uneval_config.clear()

    def get_suggestion(self) -> Configuration:
        return self.get_suggestions(batch_size=1)[0]

    def get_suggestions(self, batch_size=None) -> List[Configuration]:
        if batch_size is None:
            batch_size = self.batch_size
        # if self.__class__.__name__ == 'RegularizedEAAdvisor':
        #     print('rea batch size', batch_size)
        # if self.__class__.__name__ == 'SAEA_Advisor':
        #     print('saea batch size', batch_size)

        if len(self.cached_config) < batch_size:
            self.gen(max(self.required_evaluation_count, batch_size))

        batch_size = min(batch_size, len(self.cached_config))

        # if self.__class__.__name__ == 'RegularizedEAAdvisor':
        #     print('new batch size', len(self.cached_config))

        res = self.cached_config[:batch_size]
        self.cached_config = self.cached_config[batch_size:]
        self.uneval_config.extend(res)
        return res

    def remove_uneval(self, configs: List[Configuration]):
        for c in configs:
            if c in self.uneval_config:
                self.uneval_config.remove(c)

    def update_observation(self, observation: Observation):
        self.update_observations([observation])

    def update_observations(self, observations: List[Observation]):

        for observation in observations:
            self.history.update_observation(observation)

            pop = as_individual(observation)

            found = pop.config in self.uneval_config
            if found:
                self.uneval_config.remove(pop.config)

            if pop is not None:
                if self.keep_unexpected_population or found:
                    self.next_population.append(pop)

            if self.auto_step and self.strict_auto_step:
                if self.could_sel():
                    self.sel()

        if not self.strict_auto_step and self.auto_step:
            if self.could_sel():
                self.sel()
