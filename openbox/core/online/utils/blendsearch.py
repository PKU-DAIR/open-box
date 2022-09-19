import abc
from typing import List

import numpy as np
from ConfigSpace import ConfigurationSpace, Configuration, CategoricalHyperparameter, OrdinalHyperparameter
from ConfigSpace.hyperparameters import NumericalHyperparameter

from openbox.core.online.utils.cfo import CFO
from openbox.core.online.utils.flow2 import FLOW2
from openbox.core.online.utils.random import RandomSearch
from openbox.core.online.utils.base_searcher import Searcher, almost_equal
from openbox.utils.util_funcs import check_random_state
from openbox.utils.logging_utils import get_logger
from openbox.utils.history_container import HistoryContainer, MOHistoryContainer
from openbox.utils.constants import MAXINT, SUCCESS
from openbox.core.base import Observation


class SearchPiece:
    def __init__(self, searcher: Searcher,
                 perf,
                 cost):
        self.perf = perf
        self.cost = cost
        self.config = None
        self.search_method = searcher


class BlendSearchAdvisor(abc.ABC):
    def __init__(self, config_space: ConfigurationSpace,
                 dead_line=0,
                 globalsearch=RandomSearch,
                 localsearch=CFO,
                 num_constraints=0,
                 batch_size=1,
                 output_dir='logs',
                 task_id='default_task_id',
                 random_state=None):

        # System Settings.
        self.rng = check_random_state(random_state)
        self.output_dir = output_dir
        self.logger = get_logger(self.__class__.__name__)

        # Objectives Settings
        self.u = 1.5
        self.v = 1.0
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

        # init history container
        self.history_container = HistoryContainer(task_id, self.num_constraints, config_space=self.config_space)

        # Init
        self.cur = None
        self.time_used = 0
        self.x0 = self.sample_random_config()
        self.globals = None
        self.locals = []
        self.cur_cnt = 0
        self.max_locals = 10
        self.max_cnt = int(self.max_locals * 0.7)

    def get_suggestion(self):
        next_config = None
        if self.globals is None:
            self.globals = SearchPiece(self.GlobalSearch(self.config_space, self.x0), -MAXINT, None)
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
        pass
        config = observation.config
        perf = observation.objs[0]
        self.running_configs.remove(config)
        self.cur.perf = perf
        self.cur.search_method.update_observation(observation)
        self.merge_piece()

        return self.history_container.update_observation(observation)

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
                self.logger.warning('Cannot sample non duplicate configuration after %d iterations.' % max_sample_cnt)
                break
        return config

    def get_history(self):
        return self.history_container

    def select_piece(self):
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
        return len(self.locals) < self.max_locals

    def create_piece(self, config: Configuration):
        self.locals.append(SearchPiece(self.LocalSearch(self.config_space, config),
                                       -MAXINT, None))

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
            return self.u * s.perf - self.v * s.cost

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
                arr[i] = self.rng.randint(0, hp_type.get_size() - 1)
            elif isinstance(hp_type, NumericalHyperparameter):
                arr[i] = min(arr[i] + d[i], 1.0)

        ret = Configuration(self.config_space, vector=arr)
        if ret in self.all_configs:
            if recu > 100:
                self.logger.warning('Cannot sample non duplicate configuration after %d iterations.' % 100)
            else:
                ret = self.next(config_a, recu + 1)
        return ret
