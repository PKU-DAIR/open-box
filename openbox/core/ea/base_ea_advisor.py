import abc
import collections
import random
from typing import Union, Dict, List, Optional

from ConfigSpace import ConfigurationSpace, Configuration

from openbox import logger
from openbox.utils.util_funcs import check_random_state, deprecate_kwarg
from openbox.utils.history import Observation, History
from openbox.utils.constants import MAXINT, SUCCESS


class EAAdvisor(abc.ABC):
    """
    This is the base class for all advisors using evolutionary algorithms.
    An instance of this class may be used as an advisor somewhere else.
    This is an abstract class. Define a subclass of this to implement an advisor.
    """

    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    def __init__(self, config_space: ConfigurationSpace,
                 num_objectives=1,
                 num_constraints=0,
                 population_size=30,
                 optimization_strategy='ea',
                 batch_size=1,
                 output_dir='logs',
                 task_id='OpenBox',
                 random_state=None):

        # System Settings.

        self.rng = check_random_state(random_state)
        self.output_dir = output_dir

        # Objectives Settings
        self.num_objectives = num_objectives
        self.num_constraints = num_constraints
        self.config_space = config_space
        self.config_space_seed = self.rng.randint(MAXINT)
        self.config_space.seed(self.config_space_seed)

        # Init parallel settings
        self.batch_size = batch_size
        self.init_num = batch_size  # for compatibility in pSMBO
        self.running_configs = list()

        # Basic components in Advisor.
        # The parameter should be removed. Keep it here to avoid compatibility issues.
        self.optimization_strategy = optimization_strategy

        # Start initialization for EA variables.
        self.all_configs = set()
        self.age = 0
        self.population: List[Union[Dict, Individual]] = list()
        self.population_size = population_size
        assert self.population_size is not None

        # init history
        self.history = History(
            task_id=task_id, num_objectives=num_objectives, num_constraints=num_constraints, config_space=config_space,
            ref_point=None, meta_info=None,  # todo: add ref_point, meta_info
        )

    def get_suggestion(self):
        """
        Abstract. An advisor must implement this.
        Call this to get a suggestion from the advisor.
        The caller should evaluate this configuration and then call update_observation to send the result back.
        """
        raise NotImplementedError

    def update_observation(self, observation: Observation):
        """
        Abstract. An advisor must implement this.
        Call this to send the result back to advisor.
        It should be guaranteed that the configuration evaluated in this observation is got by calling
        get_suggestion earlier on the same advisor.
        """
        raise NotImplementedError

    def get_suggestions(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        return [self.get_suggestion() for i in range(batch_size)]

    def update_observations(self, observations: List[Observation]):
        return [self.update_observation(o) for o in observations]

    def sample_random_config(self, excluded_configs=None):
        if excluded_configs is None:
            excluded_configs = set()

        sample_cnt = 0
        max_sample_cnt = 1000
        while True:
            config = self.config_space.sample_configuration()
            sample_cnt += 1
            if config not in excluded_configs:
                break
            if sample_cnt >= max_sample_cnt:
                logger.warning('Cannot sample non duplicate configuration after %d iterations.' % max_sample_cnt)
                break
        return config

    def get_history(self):
        return self.history


class Individual:

    def __init__(self,
                 config: Configuration,
                 perf: Union[float, List[float]],
                 constraints_satisfied: bool = True,
                 data: Optional[Dict] = None,
                 **kwargs):

        self.config = config
        if isinstance(perf, float):
            self.dim = 1
        else:
            self.dim = len(perf)

        self.perf = perf
        self.constraints_satisfied = constraints_satisfied

        self.data = kwargs
        if data is not None:
            for x in data:
                self.data[x] = data[x]

    def perf_1d(self):
        assert self.dim == 1
        return self.perf if isinstance(self.perf, float) else self.perf[0]

    # For compatibility
    def __getitem__(self, item):
        if item == 'config':
            return self.config
        elif item == 'perf':
            return self.perf
        elif item == 'constraints_satisfied':
            return self.constraints_satisfied
        else:
            return self.data[item]


def as_individual(observation: Observation, allow_constraint=True) -> Optional[Individual]:
    config = observation.config
    constraint = constraint_check(observation.constraints) and observation.trial_state == SUCCESS
    if not allow_constraint and not constraint:
        return None
    perf = observation.objectives

    return Individual(config=config, constraints_satisfied=constraint, perf=perf)


def pareto_sort(population: List[Individual],
                selection_strategy='random', ascending=False) -> List[Individual]:
    t = pareto_best(population, count_ratio=1.0, selection_strategy=selection_strategy)
    if ascending:
        t.reverse()
    return t


def pareto_best(population: List[Individual],
                count: Optional[int] = None,
                count_ratio: Optional[float] = None,
                selection_strategy='random') -> List[Individual]:
    assert not (count is None and count_ratio is None)
    assert selection_strategy in ['random']

    if count is None:
        count = max(1, int(len(population) * count_ratio))

    remain = [x for x in population]

    if remain[0].dim == 1:
        remain.sort(key=lambda a: a.perf_1d())
        return remain[:count]

    res = []
    while count > 0:
        front = pareto_frontier(remain)
        assert len(front) > 0
        if selection_strategy == 'random':
            random.shuffle(front)
        if count >= len(front):
            res.extend(front)
            remain = [x for x in remain if x not in front]
            count -= len(front)
        else:
            res.extend(front[:count])
            count = 0

    return res


def pareto_layers(population: List[Individual]) -> List[List[Individual]]:
    remain = [x for x in population]

    res = []
    while remain:
        front = pareto_frontier(remain)
        assert len(front) > 0
        res.append(front)
        remain = [x for x in remain if x not in front]

    return res


# Naive Implementation
def pareto_frontier(population: List[Individual]) -> List[Individual]:
    if isinstance(population[0].perf, float):
        return [x for x in population if
                not [y for y in population if y.perf < x.perf]]
    return [x for x in population if
            not [y for y in population if not [i for i in range(len(x.perf)) if y.perf[i] >= x.perf[i]]]]


def constraint_check(constraint, positive_numbers=False) -> bool:
    if constraint is None:
        return True
    elif isinstance(constraint, bool):
        return constraint
    elif isinstance(constraint, float) or isinstance(constraint, int):
        return constraint >= 0 if positive_numbers else constraint <= 0
    elif isinstance(constraint, collections.Iterable):
        return not [x for x in constraint if not constraint_check(x, positive_numbers)]
    else:
        return bool(constraint)
