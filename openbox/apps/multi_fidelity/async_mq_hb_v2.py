# License: MIT

import time
import numpy as np
from math import log, ceil
from openbox.apps.multi_fidelity.async_mq_base_facade import async_mqBaseFacade
from openbox.apps.multi_fidelity.utils import WAITING, RUNNING, COMPLETED, PROMOTED
from openbox.apps.multi_fidelity.utils import sample_configuration

from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.constants import MAXINT


class async_mqHyperband_v2(async_mqBaseFacade):
    """
    The implementation of Asynchronous Hyperband with promotion cycle
    """
    def __init__(self, objective_func,
                 config_space: ConfigurationSpace,
                 R,
                 eta=3,
                 skip_outer_loop=0,
                 random_state=1,
                 method_id='mqAsyncHyperband_new',
                 restart_needed=True,
                 time_limit_per_trial=600,
                 runtime_limit=None,
                 ip='',
                 port=13579,
                 authkey=b'abc',):
        max_queue_len = 3 * R   # conservative design
        super().__init__(objective_func, method_name=method_id,
                         restart_needed=restart_needed, time_limit_per_trial=time_limit_per_trial,
                         runtime_limit=runtime_limit,
                         max_queue_len=max_queue_len, ip=ip, port=port, authkey=authkey)

        self.seed = random_state
        self.config_space = config_space
        self.config_space.seed(self.seed)

        self.R = R      # Maximum iterations per configuration
        self.eta = eta  # Define configuration downsampling rate (default = 3)
        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.R))
        self.B = (self.s_max + 1) * self.R

        self.incumbent_configs = list()
        self.incumbent_perfs = list()

        self.skip_outer_loop = skip_outer_loop

        self.brackets = None
        self.all_rung_list = None
        self.create_brackets()

        # construct hyperband iteration list for initial configs
        self.hb_bracket_id = 0  # index the chosen bracket in self.hb_bracket_list
        self.hb_bracket_list = list()  # record iteration lists of all brackets
        self.hb_iter_id = 0  # index the current chosen n_iteration in self.hb_iter_list
        self.hb_iter_list = list()  # record current iteration list
        self.create_hb_iter_list()

    def create_brackets(self):
        """
        bracket : list of rungs
        rung: {
            'rung_id': rung id (the lowest rung is 0),
            'bracket_id': bracket id that the rung belongs to,
            'n_iteration': iterations (resource) per config for evaluation,
            'promotion_cycle': number of configs to activate promotion,
            'jobs': list of [job_status, config, perf, extra_conf],
            'configs': set of all configs in the rung,
            'num_promoted': number of promoted configs in the rung,
        }
        job_status: RUNNING, COMPLETED, PROMOTED
        """
        self.brackets = list()
        self.all_rung_list = list()
        for s in reversed(range(self.skip_outer_loop, self.s_max + 1)):
            # Initial number of configurations
            n = int(ceil(self.B / self.R / (s + 1) * self.eta ** s))
            # Initial number of iterations per config
            r = self.R * self.eta ** (-s)

            bracket_id = self.s_max - s
            bracket = list()
            for i in range(s + 1):
                n_iteration = r * self.eta ** (i)
                promotion_cycle = int(n * self.eta ** (-i))
                rung = dict(
                    rung_id=i,
                    bracket_id=bracket_id,
                    n_iteration=n_iteration,
                    promotion_cycle=promotion_cycle,
                    jobs=list(),
                    configs=set(),
                    num_promoted=0,
                )
                bracket.append(rung)
                self.all_rung_list.append(rung)
            self.brackets.append(bracket)

        self.all_rung_list.sort(key=lambda rung: (-rung['n_iteration'], rung['bracket_id']))    # sort by n_iteration

        self.logger.info('Init brackets: %s.' % str(self.brackets))
        self.logger.info('Init all_rung_list: %s.' % str(self.all_rung_list))

    def create_hb_iter_list(self):
        for s in reversed(range(self.skip_outer_loop, self.s_max + 1)):
            # Initial number of configurations
            n = int(ceil(self.B / self.R / (s + 1) * self.eta ** s))
            # Initial number of iterations per config
            r = self.R * self.eta ** (-s)
            # construct iteration list
            self.hb_bracket_list.append([r] * n)
        self.hb_iter_list = self.hb_bracket_list[0]
        self.logger.info('hyperband iteration lists of all brackets: %s. init bracket: %s.'
                         % (self.hb_bracket_list, self.hb_iter_list))

    def get_job(self):
        """
        find waiting config, or sample a new config
        """
        next_config = None
        next_n_iteration = None
        next_extra_conf = None
        # find waiting config from top to bottom
        for rung in self.all_rung_list:  # all_rung_list is sorted by n_iteration
            for job_id, job in enumerate(rung['jobs']):
                job_status, config, perf, extra_conf = job
                # set running
                if job_status == WAITING:
                    next_config = config
                    next_n_iteration = rung['n_iteration']
                    next_extra_conf = extra_conf
                    # update bracket
                    self.logger.info('Running job in bracket %d rung %d: %s'
                                     % (rung['bracket_id'], rung['rung_id'], rung['jobs'][job_id]))
                    rung['jobs'][job_id][0] = RUNNING
                    break
            if next_config is not None:
                break

        # no waiting config, sample a new one
        if next_config is None:
            next_config, next_n_iteration, next_extra_conf = self.choose_next()
            # update bracket
            bracket_id = self.get_bracket_id(self.brackets, next_n_iteration)
            self.logger.info('Sample a new config: %s. Add to bracket %d.' % (next_config, bracket_id))
            new_job = [RUNNING, next_config, MAXINT, next_extra_conf]  # running perf is set to MAXINT
            self.brackets[bracket_id][0]['jobs'].append(new_job)
            self.brackets[bracket_id][0]['configs'].add(next_config)
            assert len(self.brackets[bracket_id][0]['jobs']) == len(self.brackets[bracket_id][0]['configs'])

        # print('=== brackets after get_job:', self.get_brackets_status(self.brackets))
        return next_config, next_n_iteration, next_extra_conf

    def check_promotion(self, bracket_id, rung_id):
        """
        check promotion cycle, then promote (set waiting)
        CAUTION: call this function only after update_observation() every time
        """
        bracket = self.brackets[bracket_id]
        # do not check highest rung
        if rung_id == len(bracket) - 1:
            return

        # check whether number of complete jobs reaches promotion cycle
        num_completed_promoted = len([job for job in bracket[rung_id]['jobs']
                                      if job[0] in (COMPLETED, PROMOTED)])
        if num_completed_promoted % bracket[rung_id]['promotion_cycle'] != 0:
            return

        # job: [job_status, config, perf, extra_conf]
        bracket[rung_id]['jobs'] = sorted(bracket[rung_id]['jobs'], key=lambda x: x[2])
        completed_jobs = [(job_id, job) for job_id, job in enumerate(bracket[rung_id]['jobs'])
                          if job[0] in (COMPLETED, )]

        n_set_promote = 0
        n_should_promote = bracket[rung_id + 1]['promotion_cycle']
        candidate_jobs = completed_jobs[0: n_should_promote]
        for job_id, job in candidate_jobs:
            job_status, config, perf, extra_conf = job
            if not perf < MAXINT:
                self.logger.warning('Skip promoting job (bad perf): %s' % job)
                continue
            # check if config already exists in upper rungs
            exist = False
            for i in range(rung_id + 1, len(bracket)):
                if config in bracket[i]['configs']:
                    exist = True
                    break
            if exist:
                self.logger.warning('Skip promoting job (duplicate): %s' % job)
                continue

            # promote (set waiting)
            n_set_promote += 1
            next_config = config
            next_n_iteration = bracket[rung_id + 1]['n_iteration']
            next_extra_conf = extra_conf
            # update bracket
            self.logger.info('Promote job in bracket %d rung %d: %s' %
                             (bracket_id, rung_id, bracket[rung_id]['jobs'][job_id]))
            bracket[rung_id]['jobs'][job_id][0] = PROMOTED
            bracket[rung_id]['num_promoted'] += 1
            new_job = [WAITING, next_config, MAXINT, next_extra_conf]  # running perf is set to MAXINT
            bracket[rung_id + 1]['jobs'].append(new_job)
            bracket[rung_id + 1]['configs'].add(next_config)
            assert len(bracket[rung_id + 1]['jobs']) == len(bracket[rung_id + 1]['configs'])

        if n_set_promote != n_should_promote:
            self.logger.warning('In rung %d, promote: %d, should promote: %d.'
                                % (rung_id, n_set_promote, n_should_promote))
        else:
            self.logger.info('In rung %d, promote %d configs.' % (rung_id, n_set_promote))

    def update_observation(self, config, perf, n_iteration):
        """
        update bracket and check promotion cycle
        """
        # update bracket
        updated = False
        updated_bracket_id, updated_rung_id = None, None
        for bracket_id, bracket in enumerate(self.brackets):
            rung_id = self.get_rung_id(bracket, n_iteration)
            if rung_id is None:
                # we check brackets in order. should be updated in previous bracket.
                raise ValueError('rung_id not found by n_iteration %d in bracket %d.' % (int(n_iteration), bracket_id))

            for job in bracket[rung_id]['jobs']:
                _job_status, _config, _perf, _extra_conf = job
                if _config == config:
                    if _job_status != RUNNING:
                        self.logger.warning('Job status is not RUNNING when update observation. '
                                            'There may exist duplicated configs in different brackets. '
                                            'bracket_id: %d, rung_id: %d, job: %s, observation: %s.'
                                            % (bracket_id, rung_id, job, (config, perf, n_iteration)))
                        break
                    job[0] = COMPLETED
                    job[2] = perf
                    updated = True
                    updated_bracket_id, updated_rung_id = bracket_id, rung_id
                    self.logger.info('update observation in bracket %d rung %d.' % (bracket_id, rung_id))
                    break
            if updated:
                break
        assert updated
        # print('=== bracket after update_observation:', self.get_brackets_status(self.brackets))

        if int(n_iteration) == self.R:
            if config in self.incumbent_configs:
                self.logger.warning('Duplicated config in self.incumbent_configs: %s' % config)
            else:
                self.incumbent_configs.append(config)
                self.incumbent_perfs.append(perf)

        # check promotion cycle
        self.check_promotion(updated_bracket_id, updated_rung_id)
        return

    def choose_next(self):
        """
        sample a random config. give iterations according to Hyperband strategy.
        """
        next_n_iteration = self.get_next_n_iteration()
        bracket_id = self.get_bracket_id(self.brackets, next_n_iteration)
        next_config = sample_configuration(self.config_space, excluded_configs=self.brackets[bracket_id][0]['configs'])
        next_extra_conf = {}
        return next_config, next_n_iteration, next_extra_conf

    def get_next_n_iteration(self):
        next_n_iteration = self.hb_iter_list[self.hb_iter_id]
        self.hb_iter_id += 1
        # next bracket
        if self.hb_iter_id == len(self.hb_iter_list):
            self.hb_iter_id = 0
            self.hb_bracket_id += 1
            if self.hb_bracket_id == len(self.hb_bracket_list):
                self.hb_bracket_id = 0
            self.hb_iter_list = self.hb_bracket_list[self.hb_bracket_id]
            self.logger.info('iteration list of next bracket: %s' % self.hb_iter_list)
        return next_n_iteration

    @staticmethod
    def get_bracket_id(brackets, n_iteration):
        """
        get bracket id by bracket initial resource
        """
        bracket_id = None
        for i, bracket in enumerate(brackets):
            if bracket[0]['n_iteration'] == n_iteration:
                bracket_id = i
                break
        assert bracket_id is not None
        return bracket_id

    @staticmethod
    def get_rung_id(bracket, n_iteration):
        rung_id = None
        for rung in bracket:
            if rung['n_iteration'] == n_iteration:
                rung_id = rung['rung_id']
                break
        # assert rung_id is not None    # can be None in Hyperband multiple brackets
        return rung_id

    @staticmethod
    def get_brackets_status(brackets):
        status = ''
        for bracket_id, bracket in enumerate(brackets):
            status += '\n' + '=' * 54 + '\n'
            status += 'bracket %d:\n' % bracket_id
            status += 'rung_id n_iteration PROMOTED COMPLETED RUNNING WAITING\n'
            for rung in bracket:
                rung_id = rung['rung_id']
                n_iteration = rung['n_iteration']
                jobs = rung['jobs']
                num_waiting, num_running, num_completed, num_promoted = 0, 0, 0, 0
                for _job_status, _config, _perf, _extra_conf in jobs:
                    if _job_status == WAITING:
                        num_waiting += 1
                    elif _job_status == RUNNING:
                        num_running += 1
                    elif _job_status == COMPLETED:
                        num_completed += 1
                    elif _job_status == PROMOTED:
                        num_promoted += 1
                status += '%7d %11d %8d %9d %7d %7d\n' % (rung_id, n_iteration,
                                                          num_promoted, num_completed, num_running, num_waiting)
            status += '=' * 54 + '\n'
        return status

    def get_incumbent(self, num_inc=1):
        assert (len(self.incumbent_perfs) == len(self.incumbent_configs))
        indices = np.argsort(self.incumbent_perfs)
        configs = [self.incumbent_configs[i] for i in indices[0:num_inc]]
        perfs = [self.incumbent_perfs[i] for i in indices[0: num_inc]]
        return configs, perfs
