# License: MIT

from math import ceil
from openbox.apps.multi_fidelity.utils import sample_configuration
from openbox.apps.multi_fidelity.async_mq_sh_v0 import async_mqSuccessiveHalving_v0

from openbox.utils.config_space import ConfigurationSpace


class async_mqHyperband_v0(async_mqSuccessiveHalving_v0):
    """
    The implementation of Asynchronous Hyperband (extended of ASHA)
    origin version
    """
    def __init__(self, objective_func,
                 config_space: ConfigurationSpace,
                 R,
                 eta=3,
                 skip_outer_loop=0,
                 random_state=1,
                 method_id='mqAsyncHyperband',
                 restart_needed=True,
                 time_limit_per_trial=600,
                 runtime_limit=None,
                 ip='',
                 port=13579,
                 authkey=b'abc',):
        super().__init__(objective_func, config_space, R, eta=eta,
                         random_state=random_state, method_id=method_id, restart_needed=restart_needed,
                         time_limit_per_trial=time_limit_per_trial, runtime_limit=runtime_limit,
                         ip=ip, port=port, authkey=authkey)

        self.skip_outer_loop = skip_outer_loop

        # construct hyperband iteration list for initial configs
        self.hb_bracket_id = 0          # index the chosen bracket in self.hb_bracket_list
        self.hb_bracket_list = list()   # record iteration lists of all brackets
        self.hb_iter_id = 0             # index the current chosen n_iteration in self.hb_iter_list
        self.hb_iter_list = list()      # record current iteration list
        self.B = (self.s_max + 1) * self.R
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

    def choose_next(self):
        """
        sample a random config. give iterations according to Hyperband strategy.
        """
        next_n_iteration = self.get_next_n_iteration()
        next_rung_id = self.get_rung_id(self.bracket, next_n_iteration)

        next_config = sample_configuration(self.config_space, excluded_configs=self.bracket[next_rung_id]['configs'])
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
