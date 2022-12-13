import abc
from typing import List

import numpy as np
from ConfigSpace import ConfigurationSpace, Configuration, CategoricalHyperparameter, OrdinalHyperparameter
from ConfigSpace.hyperparameters import NumericalHyperparameter

from openbox import logger
from openbox.core.generic_advisor import Advisor
from openbox.experimental.online.cfo import CFO
from openbox.experimental.online.base_online_advisor import almost_equal
from openbox.utils.util_funcs import check_random_state
from openbox.utils.history import Observation, History
from openbox.utils.constants import MAXINT


class SearchPiece:
    def __init__(self, searcher,
                 perf,
                 cost):
        self.perf = perf
        self.cost = cost
        self.config = None
        self.search_method = searcher


class BlendSearchAdvisor(abc.ABC):
    def __init__(self, config_space: ConfigurationSpace,
                 dead_line=0,
                 globalsearch=Advisor,
                 localsearch=CFO,
                 num_constraints=0,
                 batch_size=1,
                 pure=False,
                 output_dir='logs',
                 task_id='OpenBox',
                 random_state=None):

        # System Settings.
        self.rng = check_random_state(random_state)
        self.output_dir = output_dir

        # Objectives Settings
        self.u = 1.5
        self.v = 1.0
        self.pure = pure
        self.dead_line = dead_line
        self.GlobalSearch = globalsearch
        self.LocalSearch = localsearch
        self.num_constraints = num_constraints
        self.config_space = config_space
        self.config_space_seed = self.rng.randint(MAXINT)
        self.config_space.seed(self.config_space_seed)

        # Init parallel settings
        self.batch_size = batch_size
        self.init_num = batch_size  # for compatibility in pSMBO
        self.running_configs = list()
        self.all_configs = set()

        # init history
        self.history = History(
            task_id=task_id, num_objectives=1, num_constraints=num_constraints, config_space=config_space,
            ref_point=None, meta_info=None,  # todo: add meta info
        )

        # Init
        self.cur = None
        self.time_used = 0
        self.x0 = self.sample_random_config()
        self.globals = None
        self.locals = []
        self.cur_cnt = 0
        self.max_locals = 20
        self.max_cnt = int(self.max_locals * 0.7)

    def __str__(self):
        return f"BlendSearch({self.GlobalSearch.__name__}, {self.LocalSearch.__name__})"

    def make_searcher(self, searcher, args=(), kwargs=None):
        if kwargs is None:
            kwargs = dict()
        if isinstance(searcher, tuple):
            func = searcher[0]
            args = args if len(searcher) <= 1 else args + searcher[1]
            kwargs = kwargs if len(searcher) <= 2 else dict(list(kwargs.items()) + list(searcher[2].items()))

            return func(self.config_space, *args, **kwargs)
        else:
            return searcher(self.config_space, *args, **kwargs)

    def get_suggestion(self):
        next_config = None
        if self.globals is None:
            self.globals = SearchPiece(self.make_searcher(self.GlobalSearch), -np.inf, None)
            self.cur = self.globals
            next_config = self.globals.search_method.get_suggestion()
            self.globals.config = next_config
        else:
            next_piece = self.select_piece()
            if next_piece is self.globals and self.new_condition():
                self.create_piece(self.next(self.globals.config))
            self.cur = next_piece
            next_config = next_piece.search_method.get_suggestion()
            next_piece.config = next_config

        self.all_configs.add(next_config)
        self.running_configs.append(next_config)
        return next_config

    def update_observation(self, observation: Observation):
        config = observation.config
        perf = observation.objectives[0]
        self.running_configs.remove(config)
        self.cur.perf = perf
        self.cur.search_method.update_observation(observation)
        self.merge_piece()

        return self.history.update_observation(observation)

    def get_suggestions(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        return [self.get_suggestion() for _ in range(batch_size)]

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

    def select_piece(self):
        if self.pure:
            return self.globals
        if self.cur_cnt == self.max_cnt:
            self.cur_cnt = 0
            return self.globals
        ret = None
        for t in self.locals:
            if ret is None or self.valu(t) < self.valu(ret):
                ret = t
        if ret is None or self.valu(self.globals) < self.valu(ret):
            self.cur_cnt = 0
            ret = self.globals
        if ret is not self.globals:
            self.cur_cnt += 1
        return ret

    def new_condition(self):
        if len(self.locals) > self.max_locals:
            return False
        cnt = 0
        tot = len(self.locals)
        for t in self.locals:
            if self.valu(self.globals) < self.valu(t):
                cnt += 1
        return cnt >= tot // 2

    def create_piece(self, config: Configuration):
        self.locals.append(SearchPiece(self.make_searcher(self.LocalSearch, (config, )),
                                       -np.inf, None))

    def del_piece(self, s: SearchPiece):
        if s in self.locals:
            self.locals.remove(s)

    def merge_piece(self):
        need_del = []
        for t in self.locals:
            if t.search_method.is_converged():
                need_del.append(t)
        for t in need_del:
            self.del_piece(t)

        need_del = []
        for i, t in enumerate(self.locals):
            map(lambda x: need_del.append(x) if almost_equal(x.config, t.config) else None, self.locals[i + 1:])
        for t in need_del:
            self.del_piece(t)

    def valu(self, s: SearchPiece):
        if s.cost is None:
            return s.perf
        else:
            return self.u * s.perf + self.v * s.cost

    def next(self, config_a: Configuration, delta=0.05, gaussian=False, recu=0):
        arr = config_a.get_array().copy()
        d = np.random.randn(*arr.shape)
        if not gaussian:
            d = d / np.linalg.norm(d)
        d = d * delta

        # print(d)

        for i, key in enumerate(self.config_space.keys()):
            hp_type = self.config_space.get_hyperparameter(key)
            if isinstance(hp_type, CategoricalHyperparameter) or isinstance(hp_type, OrdinalHyperparameter):
                arr[i] = self.rng.randint(0, hp_type.get_size())
            elif isinstance(hp_type, NumericalHyperparameter):
                arr[i] = min(arr[i] + d[i], 1.0)

        ret = Configuration(self.config_space, vector=arr)
        if ret in self.all_configs:
            if recu > 100:
                logger.warning('Cannot sample non duplicate configuration after %d iterations.' % 100)
            else:
                ret = self.next(config_a, recu + 1)
        return ret
