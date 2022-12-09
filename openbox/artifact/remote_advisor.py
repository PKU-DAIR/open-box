# License: MIT

import time
import json
import requests
import hashlib

from openbox import logger
from openbox.utils.config_space import json as config_json
from openbox.utils.config_space import Configuration
from openbox.utils.constants import SUCCESS
from openbox.utils.util_funcs import deprecate_kwarg


class RemoteAdvisor(object):
    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    def __init__(self, config_space,
                 server_ip, port, email, password,
                 task_name='task',
                 num_objectives=1,
                 num_constraints=0,
                 sample_strategy: str = 'bo',
                 surrogate_type=None,
                 acq_type=None,
                 acq_optimizer_type='local_random',
                 max_runs=200,
                 init_strategy='random_explore_first',
                 initial_configurations=None,
                 initial_runs=3,
                 random_state=None,
                 time_limit_per_trial=300,
                 active_worker_num=1,
                 parallel_type='async',
                 ):

        self.email = email
        md5 = hashlib.md5()
        md5.update(password.encode('utf-8'))
        self.password = md5.hexdigest()

        # Store and serialize config space
        self.config_space = config_space
        config_space_json = config_json.write(config_space)

        # Check setup
        self.num_objectives = num_objectives
        self.num_constraints = num_constraints
        self.acq_type = acq_type
        self.constraint_surrogate_type = None
        self.surrogate_type = surrogate_type
        self.check_setup()

        # Set options
        if initial_configurations is not None and isinstance(initial_configurations[0], Configuration):
            initial_configurations = [config.get_dictionary() for config in initial_configurations]
        self.max_iterations = max_runs
        options = {
            'optimization_strategy': sample_strategy,
            'surrogate_type': surrogate_type,
            'acq_type': acq_type,
            'acq_optimizer_type': acq_optimizer_type,
            'init_strategy': init_strategy,
            'initial_configurations': initial_configurations,
            'initial_trials': initial_runs,
            'random_state': random_state
        }

        # Construct base url.
        self.base_url = 'http://%s:%d/bo_advice/' % (server_ip, port)

        # Register task
        res = requests.post(self.base_url + 'task_register/',
                            data={'email': self.email, 'password': self.password, 'task_name': task_name,
                                  'config_space_json': config_space_json,
                                  'num_constraints': num_constraints, 'num_objectives': num_objectives,
                                  'max_runs': self.max_iterations,
                                  'options': json.dumps(options), 'time_limit_per_trial': time_limit_per_trial,
                                  'active_worker_num': active_worker_num, 'parallel_type': parallel_type})
        res = json.loads(res.text)

        if res['code'] == 1:
            self.task_id = res['task_id']
        else:
            raise Exception('Server error %s' % res['msg'])

    def check_setup(self):
        """
        Check num_objectives, num_constraints, acq_type, surrogate_type.
        """
        assert isinstance(self.num_objectives, int) and self.num_objectives >= 1
        assert isinstance(self.num_constraints, int) and self.num_constraints >= 0

        # single objective no constraint
        if self.num_objectives == 1 and self.num_constraints == 0:
            if self.acq_type is None:
                self.acq_type = 'ei'
            assert self.acq_type in ['ei', 'eips', 'logei', 'pi', 'lcb', 'lpei', ]
            if self.surrogate_type is None:
                self.surrogate_type = 'prf'

        # multi-objective with constraints
        elif self.num_objectives > 1 and self.num_constraints > 0:
            if self.acq_type is None:
                self.acq_type = 'mesmoc2'
            assert self.acq_type in ['mesmoc', 'mesmoc2']
            if self.surrogate_type is None:
                self.surrogate_type = 'gp_rbf'
            if self.constraint_surrogate_type is None:
                if self.acq_type == 'mesmoc2':
                    self.constraint_surrogate_type = 'gp'
                else:
                    self.constraint_surrogate_type = 'gp_rbf'
            if self.acq_type == 'mesmoc' and self.surrogate_type != 'gp_rbf':
                self.surrogate_type = 'gp_rbf'
                logger.warning('Surrogate model has changed to Gaussian Process with RBF kernel '
                                    'since MESMOC is used. Surrogate_type should be set to \'gp_rbf\'.')
            if self.acq_type == 'mesmoc' and self.constraint_surrogate_type != 'gp_rbf':
                self.surrogate_type = 'gp_rbf'
                logger.warning('Constraint surrogate model has changed to Gaussian Process with RBF kernel '
                                    'since MESMOC is used. Surrogate_type should be set to \'gp_rbf\'.')

        # multi-objective no constraint
        elif self.num_objectives > 1:
            if self.acq_type is None:
                self.acq_type = 'mesmo'
            assert self.acq_type in ['mesmo', 'usemo']
            if self.surrogate_type is None:
                if self.acq_type == 'mesmo':
                    self.surrogate_type = 'gp_rbf'
                else:
                    self.surrogate_type = 'gp'
            if self.acq_type == 'mesmo' and self.surrogate_type != 'gp_rbf':
                self.surrogate_type = 'gp_rbf'
                logger.warning('Surrogate model has changed to Gaussian Process with RBF kernel '
                                    'since MESMO is used. Surrogate_type should be set to \'gp_rbf\'.')

        # single objective with constraints
        elif self.num_constraints > 0:
            if self.acq_type is None:
                self.acq_type = 'eic'
            assert self.acq_type in ['eic', 'ts']
            if self.surrogate_type is None:
                if self.acq_type == 'ts':
                    self.surrogate_type = 'gp'
                else:
                    self.surrogate_type = 'prf'
            if self.constraint_surrogate_type is None:
                self.constraint_surrogate_type = 'gp'
            if self.acq_type == 'ts' and self.surrogate_type != 'gp':
                self.surrogate_type = 'gp'
                logger.warning('Surrogate model has changed to Gaussian Process '
                                    'since TS is used. Surrogate_type should be set to \'gp\'.')

    def get_suggestion(self):
        res = requests.post(self.base_url + 'get_suggestion/',
                            data={'task_id': self.task_id})
        res = json.loads(res.text)

        if res['code'] == 1:
            config_dict = json.loads(res['res'])
            return config_dict
        else:
            raise Exception('Server error %s' % res['msg'])

    def update_observation(self, config_dict, objectives, constraints=[], trial_info={}, trial_state=SUCCESS):
        res = requests.post(self.base_url + 'update_observation/',
                            data={'task_id': self.task_id, 'config': json.dumps(config_dict),
                                  'objectives': json.dumps(objectives), 'constraints': json.dumps(constraints),
                                  'trial_state': trial_state, 'trial_info': json.dumps(trial_info)})
        res = json.loads(res.text)
        if res['code'] == 0:
            raise Exception('Server error %s' % res['msg'])

    def get_result(self):
        res = requests.post(self.base_url + 'get_result/', data={'task_id': self.task_id})
        res_dict = res.json()
        result = json.loads(res_dict.get('result'))
        history = json.loads(res_dict.get('history'))
        return result, history
