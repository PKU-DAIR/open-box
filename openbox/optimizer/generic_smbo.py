# License: MIT

import sys
import re
import time
import os
import json
import traceback
import math
from typing import List
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from openbox.optimizer.base import BOBase
from openbox.utils.constants import MAXINT, SUCCESS, FAILED, TIMEOUT
from openbox.utils.limit import time_limit, TimeoutException
from openbox.utils.util_funcs import get_result
from openbox.core.base import Observation

"""
    The objective function returns a dictionary that has --- config, constraints, objs ---.
"""


class SMBO(BOBase):
    """
    Parameters
    ----------
    objective_function : callable
        Objective function to optimize.
    config_space : openbox.space.Space
        Configuration space.
    num_constraints : int
        Number of constraints in objective function.
    num_objs : int
        Number of objectives in objective function.
    max_runs : int
        Number of optimization iterations.
    runtime_limit : int or float, optional
        Time budget for the whole optimization process. None means no limit.
    time_limit_per_trial : int or float
        Time budget for a single evaluation trial.
    advisor_type : str
        Type of advisor to produce configuration suggestion.
        - 'default' (default): Bayesian Optimization
        - 'tpe': Tree-structured Parzen Estimator
        - 'ea': Evolutionary Algorithms
        - 'random': Random Search
        - 'mcadvisor': Bayesian Optimization with Monte Carlo Sampling
    surrogate_type : str
        Type of surrogate model in Bayesian optimization.
        - 'gp' (default): Gaussian Process. Better performance for mathematical problems.
        - 'prf': Probability Random Forest. Better performance for hyper-parameter optimization (HPO).
        - 'lightgbm': LightGBM.
    acq_type : str
        Type of acquisition function in Bayesian optimization.
        For single objective problem:
        - 'ei' (default): Expected Improvement
        - 'eips': Expected Improvement per Second
        - 'logei': Logarithm Expected Improvement
        - 'pi': Probability of Improvement
        - 'lcb': Lower Confidence Bound
        For single objective problem with constraints:
        - 'eic' (default): Expected Constrained Improvement
        For multi-objective problem:
        - 'ehvi (default)': Expected Hypervolume Improvement
        - 'mesmo': Multi-Objective Max-value Entropy Search
        - 'usemo': Multi-Objective Uncertainty-Aware Search
        - 'parego': ParEGO
        For multi-objective problem with constraints:
        - 'ehvic' (default): Expected Hypervolume Improvement with Constraints
        - 'mesmoc': Multi-Objective Max-value Entropy Search with Constraints
    acq_optimizer_type : str
        Type of optimizer to maximize acquisition function.
        - 'local_random' (default): Interleaved Local and Random Search
        - 'random_scipy': L-BFGS-B (Scipy) optimizer with random starting points
        - 'scipy_global': Differential Evolution
        - 'cma_es': Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
    initial_runs : int
        Number of initial iterations of optimization.
    init_strategy : str
        Strategy to generate configurations for initial iterations.
        - 'random_explore_first' (default): Random sampled configs with maximized internal minimum distance
        - 'random': Random sampling
        - 'default': Default configuration + random sampling
        - 'sobol': Sobol sequence sampling
        - 'latin_hypercube': Latin hypercube sampling
    initial_configurations : List[Configuration], optional
        If provided, the initial configurations will be evaluated in initial iterations of optimization.
    ref_point : List[float], optional
        Reference point for calculating hypervolume in multi-objective problem.
        Must be provided if using EHVI based acquisition function.
    history_bo_data : List[OrderedDict], optional
        Historical data for transfer learning.
    logging_dir : str
        Directory to save log files.
    task_id : str
        Task identifier.
    random_state : int
        Random seed for RNG.
    """
    def __init__(self, objective_function: callable, config_space,
                 num_constraints=0,
                 num_objs=1,
                 sample_strategy: str = 'bo',
                 max_runs=200,
                 runtime_limit=None,
                 time_limit_per_trial=180,
                 advisor_type='default',
                 surrogate_type='auto',
                 acq_type='auto',
                 acq_optimizer_type='auto',
                 initial_runs=3,
                 init_strategy='random_explore_first',
                 initial_configurations=None,
                 ref_point=None,
                 history_bo_data: List[OrderedDict] = None,
                 logging_dir='logs',
                 task_id='default_task_id',
                 random_state=None,
                 advisor_kwargs: dict = None,
                 json_path=None,
                 vis_path_tmp=None,
                 vis_path=None,
                 **kwargs):

        if task_id is None:
            raise ValueError('Task id is not SPECIFIED. Please input task id first.')

        if json_path is None:
            json_path = os.path.join(os.path.abspath("."),  'bo_history')
            # raise ValueError('Json_path is not SPECIFIED. Please input json_path first, or we can not save your data.')
        self.json_path = json_path
        if not os.path.exists(self.json_path):
            os.makedirs(self.json_path)
        self.json_file_name = 'bo_history_%s.json' % task_id
        if os.path.exists(os.path.join(json_path, self.json_file_name)):
            raise ValueError('There is already a same task_id in your json_path. Please change task_id or json_path.')

        if vis_path_tmp is None:
            vis_path_tmp = os.path.join(os.path.abspath("."), 'bo_visualization_tmp')
            # raise ValueError('Json_path is not SPECIFIED. Please input json_path first, or we can not save your data.')
        self.vis_path_tmp = vis_path_tmp
        if not os.path.exists(self.vis_path_tmp):
            os.makedirs(self.vis_path_tmp)
        self.vis_file_name_tmp = 'bo_visualization_tmp_%s.html' % task_id
        if os.path.exists(os.path.join(vis_path_tmp, self.vis_file_name_tmp)):
            raise ValueError('There is already a same task_id in your vis_path_tmp. Please change task_id or vis_path_tmp.')

        if vis_path is None:
            vis_path = os.path.join(os.path.abspath("."), 'bo_visualization')
            # raise ValueError('Json_path is not SPECIFIED. Please input json_path first, or we can not save your data.')
        self.vis_path = vis_path
        if not os.path.exists(self.vis_path):
            os.makedirs(self.vis_path)
        self.vis_file_name = 'bo_visualization_%s.html' % task_id
        if os.path.exists(os.path.join(vis_path, self.vis_file_name)):
            raise ValueError('There is already a same task_id in your vis_path. Please change task_id or vis_path.')

        # generate visualization html file from template
        self.generate_html()
        pass

        self.num_objs = num_objs
        self.num_constraints = num_constraints
        self.data = []
        self.FAILED_PERF = [MAXINT] * num_objs
        super().__init__(objective_function, config_space, task_id=task_id, output_dir=logging_dir,
                         random_state=random_state, initial_runs=initial_runs, max_runs=max_runs,
                         runtime_limit=runtime_limit, sample_strategy=sample_strategy,
                         time_limit_per_trial=time_limit_per_trial, history_bo_data=history_bo_data)

        self.advisor_type = advisor_type
        advisor_kwargs = advisor_kwargs or {}
        if advisor_type == 'default':
            from openbox.core.generic_advisor import Advisor
            self.config_advisor = Advisor(config_space,
                                          num_objs=num_objs,
                                          num_constraints=num_constraints,
                                          initial_trials=initial_runs,
                                          init_strategy=init_strategy,
                                          initial_configurations=initial_configurations,
                                          optimization_strategy=sample_strategy,
                                          surrogate_type=surrogate_type,
                                          acq_type=acq_type,
                                          acq_optimizer_type=acq_optimizer_type,
                                          ref_point=ref_point,
                                          history_bo_data=history_bo_data,
                                          task_id=task_id,
                                          output_dir=logging_dir,
                                          random_state=random_state,
                                          **advisor_kwargs)
        elif advisor_type == 'mcadvisor':
            from openbox.core.mc_advisor import MCAdvisor
            self.config_advisor = MCAdvisor(config_space,
                                            num_objs=num_objs,
                                            num_constraints=num_constraints,
                                            initial_trials=initial_runs,
                                            init_strategy=init_strategy,
                                            initial_configurations=initial_configurations,
                                            optimization_strategy=sample_strategy,
                                            surrogate_type=surrogate_type,
                                            acq_type=acq_type,
                                            acq_optimizer_type=acq_optimizer_type,
                                            ref_point=ref_point,
                                            history_bo_data=history_bo_data,
                                            task_id=task_id,
                                            output_dir=logging_dir,
                                            random_state=random_state,
                                            **advisor_kwargs)
        elif advisor_type == 'tpe':
            from openbox.core.tpe_advisor import TPE_Advisor
            assert num_objs == 1 and num_constraints == 0
            self.config_advisor = TPE_Advisor(config_space, task_id=task_id, random_state=random_state,
                                              **advisor_kwargs)
        elif advisor_type == 'ea':
            from openbox.core.ea_advisor import EA_Advisor
            assert num_objs == 1 and num_constraints == 0
            self.config_advisor = EA_Advisor(config_space,
                                             num_objs=num_objs,
                                             num_constraints=num_constraints,
                                             optimization_strategy=sample_strategy,
                                             batch_size=1,
                                             task_id=task_id,
                                             output_dir=logging_dir,
                                             random_state=random_state,
                                             **advisor_kwargs)
        elif advisor_type == 'random':
            from openbox.core.random_advisor import RandomAdvisor
            self.config_advisor = RandomAdvisor(config_space,
                                                num_objs=num_objs,
                                                num_constraints=num_constraints,
                                                initial_trials=initial_runs,
                                                init_strategy=init_strategy,
                                                initial_configurations=initial_configurations,
                                                surrogate_type=surrogate_type,
                                                acq_type=acq_type,
                                                acq_optimizer_type=acq_optimizer_type,
                                                ref_point=ref_point,
                                                history_bo_data=history_bo_data,
                                                task_id=task_id,
                                                output_dir=logging_dir,
                                                random_state=random_state,
                                                **advisor_kwargs)
        else:
            raise ValueError('Invalid advisor type!')

    def run(self):
        for _ in tqdm(range(self.iteration_id, self.max_iterations)):
            if self.budget_left < 0:
                self.logger.info('Time %f elapsed!' % self.runtime_limit)
                break
            start_time = time.time()
            self.iterate(budget_left=self.budget_left)
            runtime = time.time() - start_time
            self.budget_left -= runtime

        self.visualize()
        return self.get_history()

    def iterate(self, budget_left=None):
        # get configuration suggestion from advisor
        config = self.config_advisor.get_suggestion()

        trial_state = SUCCESS
        _budget_left = int(1e10) if budget_left is None else budget_left
        _time_limit_per_trial = math.ceil(min(self.time_limit_per_trial, _budget_left))

        # only evaluate non duplicate configuration
        if config not in self.config_advisor.history_container.configurations:
            start_time = time.time()
            try:
                # evaluate configuration on objective_function within time_limit_per_trial
                args, kwargs = (config,), dict()
                timeout_status, _result = time_limit(self.objective_function,
                                                     _time_limit_per_trial,
                                                     args=args, kwargs=kwargs)
                if timeout_status:
                    raise TimeoutException(
                        'Timeout: time limit for this evaluation is %.1fs' % _time_limit_per_trial)
                else:
                    # parse result
                    objs, constraints = get_result(_result)
            except Exception as e:
                # parse result of failed trial
                if isinstance(e, TimeoutException):
                    self.logger.warning(str(e))
                    trial_state = TIMEOUT
                else:
                    self.logger.warning('Exception when calling objective function: %s' % str(e))
                    trial_state = FAILED
                objs = self.FAILED_PERF
                constraints = None

            elapsed_time = time.time() - start_time
            # update observation to advisor
            observation = Observation(
                config=config, objs=objs, constraints=constraints,
                trial_state=trial_state, elapsed_time=elapsed_time,
            )
            if _time_limit_per_trial != self.time_limit_per_trial and trial_state == TIMEOUT:
                # Timeout in the last iteration.
                pass
            else:
                self.config_advisor.update_observation(observation)
                self.save_json(observation)
        else:
            self.logger.info('This configuration has been evaluated! Skip it: %s' % config)
            history = self.get_history()
            config_idx = history.configurations.index(config)
            trial_state = history.trial_states[config_idx]
            objs = history.perfs[config_idx]
            constraints = history.constraint_perfs[config_idx] if self.num_constraints > 0 else None
            if self.num_objs == 1:
                objs = (objs,)

        self.iteration_id += 1
        # Logging.
        if self.num_constraints > 0:
            self.logger.info('Iteration %d, objective value: %s. constraints: %s.'
                             % (self.iteration_id, objs, constraints))
        else:
            self.logger.info('Iteration %d, objective value: %s.' % (self.iteration_id, objs))

        # Visualization.
        # for idx, obj in enumerate(objs):
        #     if obj < self.FAILED_PERF[idx]:
        #         self.writer.add_scalar('data/objective-%d' % (idx + 1), obj, self.iteration_id)
        return config, trial_state, constraints, objs

    def save_json(self, res: Observation):

        data_item = dict(
                task_id=self.task_id,
                iteration_id=self.iteration_id,
                config=res.config.get_dictionary(),
                objs=res.objs,
                constaints=res.constraints,
                trial_state=res.trial_state,
                cost=res.elapsed_time,
            )
        self.data.append(data_item)

        with open(os.path.join(self.json_path, self.json_file_name), 'w') as fp:
            json.dump({'data': self.data}, fp, indent=2)
        print('Save history to %s' % self.json_file_name)

        with open(os.path.join(self.json_path, 'visual_'+self.json_file_name), 'w') as fp:
            fp.write('var info=')
            json.dump({'data': self.data}, fp, indent=2)
            fp.write(';')
        print('Save history to visual_%s' % self.json_file_name)

    def load_json(self):
        with open(os.path.join(self.json_path, self.json_file_name), 'r') as fp:
            json_data = json.load(fp)

        json_data = json_data['data']

        table_list = []
        rh_config = {}
        option = {'data': [], 'schema': [], 'visualMap': {}}
        perf_list = []
        for rh in json_data:
            result = round(rh['objs'][0], 4)
            config_str = str(rh['config'])
            if len(config_str) > 35:
                config_str = config_str[1:35]
            else:
                config_str = config_str[1:-1]
            table_list.append(
                [rh['iteration_id'], result, config_str, rh['trial_state'], rh['cost']])
            rh_config[str(rh['iteration_id'])] = rh['config']
            config_values = []
            for parameter in rh['config'].keys():
                config_values.append(rh['config'][parameter])
            config_values.append(result)
            option['data'].append(config_values)
            perf_list.append(result)

        if len(json_data) > 0:
            option['schema'] = list(json_data[0]['config'].keys()) + ['perf']
            option['visualMap']['min'] = np.percentile(perf_list, 0)
            option['visualMap']['max'] = np.percentile(perf_list, 90)
            option['visualMap']['dimension'] = len(option['schema']) - 1
        else:
            option['visualMap']['min'] = 0
            option['visualMap']['max'] = 100
            option['visualMap']['dimension'] = 0

        line_data = {'min': [], 'over': [], 'scat': []}
        import sys

        min_value = sys.maxsize

        for idx, perf in enumerate(perf_list):
            if perf <= min_value:
                min_value = perf
                line_data['min'].append([idx, perf])
                line_data['scat'].append([idx, perf])
            else:
                line_data['over'].append([idx, perf])
        line_data['min'].append([len(option['data']), min_value])
        line_data['scat'].append([len(option['data']), min_value])

        return {'line_data': line_data, 'parallel_data': option, 'table_list': table_list, 'rh_config': rh_config}

    def visualize(self):
        draw_data = self.load_json()

        # task information table
        draw_data['task_inf'] = {
            'table_field': ['task_id', 'Advisor Type', 'Surrogate Type', 'max_runs', 'Time Limit Per Trial'],
            'table_data': [self.task_id, self.advisor_type, self.surrogate_type, self.max_iterations, self.time_limit_per_trial]
        }
        print(draw_data)
        from openbox.utils.visualization.visualization_for_openbox import vis_openbox
        vis_openbox(draw_data, os.path.join(self.vis_path_tmp, self.vis_file_name_tmp))

        # f = open(os.path.join(self.vis_path, "visual_template.html"),"r") 
        # l = open(os.path.join(self.vis_path, "local_"+self.vis_file_name),"w")

        # for content in f.readlines():
        #     l.write(content)

        # f.close()
        # l.close()

    def generate_html(self):
        visual_json_path = os.path.join(self.json_path, 'visual_'+self.json_file_name)
        template_path = os.path.join(os.path.abspath("."), 'bo_visualization/templates/visual_template.html')
        html_path = os.path.join(self.vis_path, self.vis_file_name)

        html_text = ''
        with open(template_path, 'r') as f:
            html_text = f.read()

        result = re.sub("<script type=\"text/javascript\" src=\"json_path\"></script>",
                        "<script type=\"text/javascript\" src=" + repr(visual_json_path) + "></script>", html_text)

        with open(html_path, "w+") as f:
            f.write(result)



