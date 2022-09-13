import abc
import collections
import random
from typing import Union, Dict, List, Optional

from ConfigSpace import ConfigurationSpace, Configuration

from .cfo import CFO
from .flow2 import FLOW2
from .base_searcher import Searcher
from openbox.utils.util_funcs import check_random_state
from openbox.utils.logging_utils import get_logger
from openbox.utils.history_container import HistoryContainer, MOHistoryContainer
from openbox.utils.constants import MAXINT, SUCCESS
from openbox.core.base import Observation

GlobalSearch = CFO
LocalSearch = FLOW2


class SearchPiece():
    def __init__(self,searcher: Searcher,
                 perf,
                 cost):
        self.perf = perf
        self.cost = cost
        self.searchmethod = searcher


def valu(s: SearchPiece):
    return s.perf


class BlendSearchAdvisor(abc.ABC):
    def __init__(self, config_space: ConfigurationSpace,
                 dead_line=0,
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
        self.dead_line = dead_line
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

    def get_suggestion(self):
        next_config = None
        if self.globals is None:
            self.globals = SearchPiece(GlobalSearch(self.config_space, self.x0), -MAXINT, None)
            self.cur = self.globals
            next_config = self.globals.searchmethod.get_suggestion()
        else:
            next_piece = self.select_piece()
            if next_piece is self.globals and self.new_condition():
                self.create_piece()
            self.cur = next_piece
            next_config = next_piece.searchmethod.get_suggestion()

        self.all_configs.add(next_config)
        self.running_configs.append(next_config)
        return next_config

    def update_observation(self, observation: Observation):
        pass
        config = observation.config
        perf = observation.objs[0]
        self.running_configs.remove(config)
        self.cur.perf = perf
        self.cur.searchmethod.update_observation(observation)
        self.merge_piece()

        return self.history_container.update_observation(observation)

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
                self.logger.warning('Cannot sample non duplicate configuration after %d iterations.' % max_sample_cnt)
                break
        return config

    def get_history(self):
        return self.history_container

    def select_piece(self):
        ret = None
        for t in self.locals:
            if ret is None or valu(t) < valu(ret):
                ret = t
        if ret is None or valu(self.globals) < valu(ret):
            ret = self.globals
        return ret

    def new_condition(self):
        return len(self.locals) < 10

    def create_piece(self):
        self.locals.append(SearchPiece(LocalSearch(self.config_space, self.globals.searchmethod.config),
                                       -MAXINT, None))

    def del_piece(self, s: SearchPiece):
        self.locals.remove(s)

    def merge_piece(self):
        return
