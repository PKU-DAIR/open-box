# License: MIT

import sys
import re
import time
import os
import json
import copy
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
            json_path = os.path.join(os.path.abspath("."), 'bo_history')
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
            raise ValueError(
                'There is already a same task_id in your vis_path_tmp. Please change task_id or vis_path_tmp.')

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

        # self.visualize()
        return self.get_history()

    def iterate(self, budget_left=None):
        self.iteration_id += 1
        # get configuration suggestion from advisor
        config = self.config_advisor.get_suggestion()

        # config_array = np.array([list(config.get_dictionary().values())])
        # predic_objs, var = self.config_advisor.surrogate_model.predict(config_array)
        # print(predic_objs, var)

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
            constaints=list(res.constraints) if self.num_constraints > 0 else None,
            trial_state=res.trial_state,
            cost=res.elapsed_time,
        )
        self.data.append(data_item)

        if self.iteration_id % 5 == 0 or self.iteration_id >= self.max_iterations:
            with open(os.path.join(self.json_path, self.json_file_name), 'w') as fp:
                json.dump({'data': self.data}, fp, indent=2)
            print('Iteration %d, Save history to %s' % (self.iteration_id, self.json_file_name))

            with open(os.path.join(self.json_path, 'visual_' + self.json_file_name), 'w') as fp:
                fp.write('var info=')
                json.dump({'data': self.process_data()}, fp, indent=2)
                fp.write(';')
            print('Iteration %d, Save history to visual_%s' % (self.iteration_id, self.json_file_name))

    def process_data(self):
        # with open(os.path.join(self.json_path, self.json_file_name), 'r') as fp:
        #     json_data = json.load(fp)

        json_data = self.data

        # Config Table data
        table_list = []
        # all the config list
        rh_config = {}
        # Parallel Data
        option = {'data': [list() for i in range(self.num_objs)], 'schema': [], 'visualMap': {}}
        # all the performance
        perf_list = [list() for i in range(self.num_objs)]
        # all the constraints
        cons_list = [list() for i in range(self.num_constraints)]
        for rh in json_data:
            results = [round(tmp, 4) for tmp in rh['objs']]
            constraints = None
            if rh['constaints']:
                constraints = [round(tmp, 4) for tmp in rh['constaints']]

            config_str = str(rh['config'])
            if len(config_str) > 35:
                config_str = config_str[1:35]
            else:
                config_str = config_str[1:-1]

            table_list.append(
                [rh['iteration_id'], results, constraints, config_str, rh['trial_state'], round(rh['cost'], 3)])

            rh_config[str(rh['iteration_id'])] = rh['config']

            config_values = []
            for parameter in rh['config'].keys():
                config_values.append(rh['config'][parameter])

            for i in range(self.num_objs):
                option['data'][i].append(config_values + [results[i]])

            for i in range(self.num_objs):
                perf_list[i].append(results[i])

            for i in range(self.num_constraints):
                cons_list[i].append(constraints[i])

        if len(json_data) > 0:
            option['schema'] = list(json_data[0]['config'].keys()) + ['perf']
            mi = float('inf')
            ma = -float('inf')
            for i in range(self.num_objs):
                mi = min(mi, np.percentile(perf_list[i], 0))
                ma = max(ma, np.percentile(perf_list[i], 90))
            option['visualMap']['min'] = mi
            option['visualMap']['max'] = ma
            option['visualMap']['dimension'] = len(option['schema']) - 1
        else:
            option['visualMap']['min'] = 0
            option['visualMap']['max'] = 100
            option['visualMap']['dimension'] = 0

        # Line Data
        line_data = [{'min': [], 'over': [], 'scat': []} for i in range(self.num_objs)]
        import sys

        for i in range(self.num_objs):
            min_value = sys.maxsize
            for idx, perf in enumerate(perf_list[i]):
                if perf <= min_value:
                    min_value = perf
                    line_data[i]['min'].append([idx, perf])
                    line_data[i]['scat'].append([idx, perf])
                else:
                    line_data[i]['over'].append([idx, perf])
            line_data[i]['min'].append([len(option['data'][i]), min_value])
            line_data[i]['scat'].append([len(option['data'][i]), min_value])

        history = self.get_history()

        # Pareto data
        pareto = dict({})
        if self.num_objs > 1:
            pareto["ref_point"] = history.ref_point
            pareto["hv"] = history.hv_data
            pareto["pareto_point"] = list(history.pareto.values())
            pareto["all_points"] = history.perfs

        # Importance data
        history_dict = history.get_importance(method='shap', return_allvalue=True)
        importance_dict = history_dict['importance_dict']
        con_importance_dict = history_dict['con_importance_dict']
        importance = {
            'X': list(history_dict['X']),
            'x': list(importance_dict.keys()),
            'data': dict(),
            'con_data': dict(),
            'obj_shap_value': history_dict['obj_shap_value'],
            'con_shap_value': history_dict['con_shap_value'],
            # 'con_importance_dict':history_dict['con_importance_dict']
        }

        for key, value in con_importance_dict.items():
            for i in range(len(value)):
                y_name = 'con-value-' + str(i + 1)
                if y_name not in importance['con_data']:
                    importance['con_data'][y_name] = list()
                importance['con_data'][y_name].append(value[i])

        for key, value in importance_dict.items():
            for i in range(len(value)):
                y_name = 'opt-value-' + str(i + 1)
                if y_name not in importance['data']:
                    importance['data'][y_name] = list()
                importance['data'][y_name].append(value[i])

        draw_data = {
            'num_objs': self.num_objs, 'num_constraints': self.num_constraints,
            'line_data': line_data,
            'cons_line_data': [[[idx, con] for idx, con in enumerate(c_l)] for c_l in cons_list],
            'parallel_data': option, 'table_list': table_list, 'rh_config': rh_config,
            'importance_data': importance,
            'pareto_data': pareto,
            'task_inf': {
                'table_field': ['task_id', 'Advisor Type', 'Surrogate Type', 'max_runs',
                                'Time Limit Per Trial'],
                'table_data': [self.task_id, self.advisor_type, self.surrogate_type, self.max_iterations,
                               self.time_limit_per_trial]
            },
            'pre_label_data': None,
            'grade_data': None,
            'cons_pre_label_data': None
        }

        if self.iteration_id >= self.max_iterations:
            draw_data['pre_label_data'], draw_data['grade_data'] = self.verify_surrogate()
            if self.num_constraints > 0:
                draw_data['cons_pre_label_data'] = self.cons_verify_surrogate()

        return draw_data

    def verify_surrogate(self):
        his = self.config_advisor.history_container
        # only get successful observations
        confs = [his.configurations[i] for i in range(len(his.configurations)) if i not in his.failed_index]
        if self.num_objs == 1:
            perfs = [his.successful_perfs]
        else:
            perfs = [[per[i] for per in his.successful_perfs] for i in range(self.num_objs)]

        N = len(perfs[0])
        if len(confs) != N or N == 0:
            print("No equal!")
            return None, None

        from openbox.utils.config_space.util import convert_configurations_to_array
        X_all = convert_configurations_to_array(confs)
        Y_all = [np.array(perfs[i], dtype=np.float64) for i in range(self.num_objs)]
        if self.num_objs == 1:
            models = [copy.deepcopy(self.config_advisor.surrogate_model)]
        else:
            models = copy.deepcopy(self.config_advisor.surrogate_model)

        # leave one out validation
        pre_perfs = [list() for i in range(self.num_objs)]

        for i in range(self.num_objs):
            for j in range(N):
                X = np.concatenate((X_all[:j, :], X_all[j + 1:, :]), axis=0)
                Y = np.concatenate((Y_all[i][:j], Y_all[i][j + 1:]))
                # 如果是多目标，那么这就会是一个模型列表
                tmp_model = copy.deepcopy(models[i])
                tmp_model.train(X, Y)

                test_X = X_all[j:j + 1, :]
                pre_mean, pre_var = tmp_model.predict(test_X)
                # 这里多目标可能要改
                pre_perfs[i].append(pre_mean[0][0])

        ranks = [[0] * N for i in range(self.num_objs)]
        pre_ranks = [[0] * N for i in range(self.num_objs)]
        for i in range(self.num_objs):
            tmp = np.argsort(perfs[i]).astype(int)
            pre_tmp = np.argsort(pre_perfs[i]).astype(int)

            for j in range(N):
                ranks[i][tmp[j]] = j + 1
                pre_ranks[i][pre_tmp[j]] = j + 1

        min1 = float('inf')
        min2 = float('inf')
        max1 = -float('inf')
        max2 = -float('inf')
        for i in range(self.num_objs):
            min1 = min(min1, round(min(min(min(pre_perfs[i]), min(perfs[i])), 0), 3))
            min2 = min(min2, round(min(min(min(pre_ranks[i]), min(ranks[i])), 0), 3))
            max1 = max(max1, round(max(min(pre_perfs[i]), max(perfs[i])), 3))
            max2 = max(max2, round(max(min(pre_ranks[i]), max(ranks[i])), 3))
        return {
                   'data': [[[pre_perfs[i][j], perfs[i][j]] for j in range(len(perfs[i]))] for i in
                            range(self.num_objs)],
                   'min': min1,
                   'max': max1
               }, \
               {
                   'data': [[[pre_ranks[i][j], ranks[i][j]] for j in range(len(ranks[i]))] for i in
                            range(self.num_objs)],
                   'min': min2,
                   'max': max2
               }

    def cons_verify_surrogate(self):
        his = self.config_advisor.history_container
        confs = his.configurations
        # 每个实例的cons都是一个列表
        cons_perfs = [[tmp[i] for tmp in his.constraint_perfs] for i in range(self.num_constraints)]
        print(cons_perfs)

        N = len(cons_perfs[0])
        if len(confs) != N or N == 0:
            print("No equal!")
            return None, None

        from openbox.utils.config_space.util import convert_configurations_to_array
        X_all = convert_configurations_to_array(confs)
        Y_all = [np.array(cons_perfs[i], dtype=np.float64) for i in range(self.num_constraints)]
        models = copy.deepcopy(self.config_advisor.constraint_models)

        # leave one out validation
        pre_perfs = [list() for i in range(self.num_constraints)]

        for i in range(self.num_constraints):
            for j in range(N):
                X = np.concatenate((X_all[:j, :], X_all[j + 1:, :]), axis=0)
                Y = np.concatenate((Y_all[i][:j], Y_all[i][j + 1:]))
                # 如果是多目标，那么这就会是一个模型列表
                tmp_model = copy.deepcopy(models[i])
                tmp_model.train(X, Y)

                test_X = X_all[j:j + 1, :]
                pre_mean, pre_var = tmp_model.predict(test_X)
                # 这里多目标可能要改
                pre_perfs[i].append(pre_mean[0][0])

        print(pre_perfs)
        min1 = float('inf')
        max1 = -float('inf')
        for i in range(self.num_objs):
            min1 = min(min1, round(min(min(min(pre_perfs[i]), min(cons_perfs[i])), 0), 3))
            max1 = max(max1, round(max(min(pre_perfs[i]), max(cons_perfs[i])), 3))

        return {
            'data': [[[pre_perfs[i][j], cons_perfs[i][j]] for j in range(len(cons_perfs[i]))] for i in range(self.num_constraints)],
            'min': min1,
            'max': max1
        }

    def visualize(self):
        draw_data = self.load_json()

        # print(draw_data)
        from openbox.utils.visualization.visualization_for_openbox import vis_openbox
        vis_openbox(draw_data, os.path.join(self.vis_path_tmp, self.vis_file_name_tmp))

    def generate_html(self):
        static_path = os.path.join(os.path.abspath("."), 'bo_visualization/static')
        template_path = os.path.join(os.path.abspath("."), 'bo_visualization/templates/visual_template.html')
        html_path = os.path.join(self.vis_path, self.vis_file_name)

        with open(template_path, 'r', encoding='utf-8') as f:
            html_text = f.read()

        link1_path = os.path.join(static_path, 'vendor/bootstrap/css/bootstrap.min.css')
        html_text = re.sub("<link rel=\"stylesheet\" href=\"../static/vendor/bootstrap/css/bootstrap.min.css\">",
                           "<link rel=\"stylesheet\" href=" + repr(link1_path) + ">", html_text)

        link2_path = os.path.join(static_path, 'css/style.default.css')
        html_text = re.sub("<link rel=\"stylesheet\" href=\"../static/css/style.default.css\" id=\"theme-stylesheet\">",
                           "<link rel=\"stylesheet\" href=" + repr(link2_path) + " id=\"theme-stylesheet\">", html_text)

        link3_path = os.path.join(static_path, 'css/custom.css')
        html_text = re.sub("<link rel=\"stylesheet\" href=\"../static/css/custom.css\">",
                           "<link rel=\"stylesheet\" href=" + repr(link3_path) + ">", html_text)

        visual_json_path = os.path.join(self.json_path, 'visual_' + self.json_file_name)
        html_text = re.sub("<script type=\"text/javascript\" src='json_path'></script>",
                           "<script type=\"text/javascript\" src=" + repr(visual_json_path) + "></script>", html_text)

        script1_path = os.path.join(static_path, 'vendor/jquery/jquery.min.js')
        html_text = re.sub("<script src=\"../static/vendor/jquery/jquery.min.js\"></script>",
                           "<script src=" + repr(script1_path) + "></script>", html_text)

        script2_path = os.path.join(static_path, 'vendor/bootstrap/js/bootstrap.bundle.min.js')
        html_text = re.sub("<script src=\"../static/vendor/bootstrap/js/bootstrap.bundle.min.js\"></script>",
                           "<script src=" + repr(script2_path) + "></script>", html_text)

        script3_path = os.path.join(static_path, 'vendor/jquery.cookie/jquery.cookie.js')
        html_text = re.sub("<script src=\"../static/vendor/jquery.cookie/jquery.cookie.js\"></script>",
                           "<script src=" + repr(script3_path) + "></script>", html_text)

        script4_path = os.path.join(static_path, 'vendor/datatables/js/datatables.js')
        html_text = re.sub("<script src=\"../static/vendor/datatables/js/datatables.js\"></script>",
                           "<script src=" + repr(script4_path) + "></script>", html_text)

        script5_path = os.path.join(static_path, 'js/echarts.min.js')
        html_text = re.sub("<script src=\"../static/js/echarts.min.js\"></script>",
                           "<script src=" + repr(script5_path) + "></script>", html_text)

        script6_path = os.path.join(static_path, 'js/common.js')
        html_text = re.sub("<script src=\"../static/js/common.js\"></script>",
                           "<script src=" + repr(script6_path) + "></script>", html_text)

        with open(html_path, "w+") as f:
            f.write(html_text)
