import os
import re
from datetime import datetime
import json
import math
import copy
from typing import List, Union
import numpy as np
from openbox import logger
from openbox.utils.history import History
from openbox.visualization.base_visualizer import BaseVisualizer
from openbox.surrogate.base.base_model import AbstractModel


class HTMLVisualizer(BaseVisualizer):
    _default_advanced_analysis_options = dict(
        importance_update_interval=5,
    )

    def __init__(
            self,
            logging_dir: str,
            history: History,
            auto_open_html: bool = False,
            advanced_analysis: bool = False,
            advanced_analysis_options: dict = None,
            advisor_type: str = None,
            surrogate_type: str = None,
            max_iterations: int = None,
            time_limit_per_trial: int = None,
            surrogate_model: Union[AbstractModel, List[AbstractModel]] = None,
            constraint_models: List[AbstractModel] = None,
    ):
        super().__init__()
        assert isinstance(logging_dir, str) and logging_dir != ''
        task_id = history.task_id
        self.output_dir = os.path.join(logging_dir, "history/%s/" % task_id)
        self.output_dir = os.path.abspath(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.auto_open_html = auto_open_html

        self.advanced_analysis = advanced_analysis
        if advanced_analysis_options is None:
            advanced_analysis_options = dict()
        self.advanced_analysis_options = self._default_advanced_analysis_options.copy()
        self.advanced_analysis_options.update(advanced_analysis_options)
        self._cache_advanced_data = dict()

        self.history = history
        self.meta_data = {
            'task_id': task_id,
            'advisor_type': advisor_type,
            'surrogate_type': surrogate_type,
            'max_iterations': max_iterations,
            'time_limit_per_trial': time_limit_per_trial,
        }
        self.surrogate_model = surrogate_model  # todo: if model is altered, this will not be updated
        self.constraint_models = constraint_models
        self.timestamp = None
        self.html_path = None
        self.displayed_html_path = None
        self.json_path = None

        if self.advanced_analysis:
            self.check_dependency()

    def setup(self, open_html=None):
        self.timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        task_id = self.meta_data['task_id']
        self.html_path = os.path.join(self.output_dir, "%s_%s.html" % (task_id, self.timestamp))
        self.displayed_html_path = 'file://' + self.html_path
        self.json_path = os.path.join(self.output_dir, "visualization_data_%s_%s.json" % (task_id, self.timestamp))
        self.generate_html()  # todo: check file conflict
        if open_html is None:
            open_html = self.auto_open_html
        if open_html:
            self.open_html()

    def update(self, update_importance=None, verify_surrogate=None):
        iter_id = len(self.history)
        max_iter = self.meta_data['max_iterations'] or np.inf
        if update_importance is None:
            if not self.advanced_analysis:
                update_importance = False
            else:
                update_interval = self.advanced_analysis_options['importance_update_interval']
                update_importance = iter_id and ((iter_id % update_interval == 0) or (iter_id >= max_iter))
        if verify_surrogate is None:
            verify_surrogate = False if not self.advanced_analysis else (iter_id >= max_iter)
        self.save_visualization_data(update_importance=update_importance, verify_surrogate=verify_surrogate)

        if iter_id == max_iter:
            logger.info('Please open the html file to view visualization result: %s' % self.displayed_html_path)

    def visualize(self, open_html=True, show_importance=False, verify_surrogate=False):
        if show_importance:
            self.check_dependency()
        self.setup(open_html=False)
        self.update(update_importance=show_importance, verify_surrogate=verify_surrogate)
        if open_html:
            self.open_html()

    def check_dependency(self):
        try:
            import shap
            import lightgbm
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                'Please install shap and lightgbm to use SHAP feature importance analysis. '
                'Run "pip install shap lightgbm"!'
            ) from e

    def save_visualization_data(self, update_importance=False, verify_surrogate=False):
        try:
            # basic data
            draw_data = self.generate_basic_data()

            # advanced data
            # importance data
            importance = self._cache_advanced_data.get('importance')
            if update_importance:
                importance = self.generate_importance_data(method='shap')
                self._cache_advanced_data['importance'] = importance
            draw_data['importance_data'] = importance
            # verify surrogate data
            if verify_surrogate:
                pre_label_data, grade_data, cons_pre_label_data = self.generate_verify_surrogate_data()
                draw_data['pre_label_data'] = pre_label_data
                draw_data['grade_data'] = grade_data
                draw_data['cons_pre_label_data'] = cons_pre_label_data

            # save data to json file
            with open(self.json_path, 'w') as fp:
                fp.write('var info=')
                json.dump({'data': draw_data}, fp, indent=2)
                fp.write(';')
        except Exception:
            logger.exception('Failed to save visualization data!')

    def generate_basic_data(self):
        # Config Table data
        table_list = []
        # all the config list
        rh_config = {}
        # Parallel Data
        option = {'data': [list() for i in range(self.history.num_objectives)], 'schema': [], 'visualMap': {}}
        # all the performance
        perf_list = [list() for i in range(self.history.num_objectives)]
        # all the constraints, A[i][j]: value of constraint i, configuration j
        cons_list = [list() for i in range(self.history.num_constraints)]
        # A[i][j]: value of configuration i, constraint j
        cons_list_rev = list()

        # todo: use observations
        # todo: check if has invalid value
        # all_objectives = self.history.get_objectives(transform='none', warn_invalid_value=False)
        # all_constraints = self.history.get_constraints(transform='none', warn_invalid_value=False)
        all_objectives = self.history.objectives
        all_constraints = self.history.constraints
        all_config_dicts = self.history.get_config_dicts()
        all_trial_states = self.history.trial_states
        all_elapsed_times = self.history.elapsed_times
        for idx in range(len(self.history)):
            results = [round(v, 4) for v in all_objectives[idx]]
            constraints = None
            if self.history.num_constraints > 0:
                constraints = [round(v, 4) for v in all_constraints[idx]]
                cons_list_rev.append(constraints)

            config_dic = all_config_dicts[idx]
            config_str = str(config_dic)
            if len(config_str) > 35:
                config_str = config_str[1:35]
            else:
                config_str = config_str[1:-1]

            table_list.append(
                [idx + 1, results, constraints, config_str, all_trial_states[idx],
                 round(all_elapsed_times[idx], 3)])

            rh_config[str(idx + 1)] = config_dic

            config_values = []
            for parameter in config_dic.keys():
                config_values.append(config_dic[parameter])

            for i in range(self.history.num_objectives):
                option['data'][i].append(config_values + [results[i]])

            for i in range(self.history.num_objectives):
                perf_list[i].append(results[i])

            for i in range(self.history.num_constraints):
                cons_list[i].append(constraints[i])

        if len(self.history) > 0:
            parameters = self.history.get_config_space().get_hyperparameter_names()
            option['schema'] = list(parameters) + ['perf']
            mi = float('inf')
            ma = -float('inf')
            for i in range(self.history.num_objectives):
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
        # ok: fits the constraint, and at the bottom.
        # no：not fits the constraint.
        # other：fits the constraint, not at the bottom
        line_data = [{'ok': [], 'no': [], 'other': []} for i in range(self.history.num_objectives)]

        for i in range(self.history.num_objectives):
            min_value = float("inf")
            for idx, perf in enumerate(perf_list[i]):
                if self.history.num_constraints > 0 and np.any(
                        [cons_list_rev[idx][k] > 0 for k in range(self.history.num_constraints)]):
                    line_data[i]['no'].append([idx, perf])
                    continue
                if perf <= min_value:
                    min_value = perf
                    line_data[i]['ok'].append([idx, perf])
                else:
                    line_data[i]['other'].append([idx, perf])
            line_data[i]['ok'].append([len(option['data'][i]), min_value])

        # Pareto data
        # todo: if ref_point is None?
        # todo: if has invalid value?
        pareto = dict()
        if self.history.num_objectives > 1:
            pareto["ref_point"] = self.history.ref_point
            hypervolumes = self.history.compute_hypervolume(data_range='all')
            pareto["hv"] = [[idx, round(v, 3)] for idx, v in enumerate(hypervolumes)]
            pareto["pareto_point"] = self.history.get_pareto_front().tolist()
            pareto["all_points"] = self.history.get_objectives(transform='none', warn_invalid_value=True).tolist()

        draw_data = {
            'num_objectives': self.history.num_objectives, 'num_constraints': self.history.num_constraints,
            'advance': self.advanced_analysis,
            'line_data': line_data,
            'cons_line_data': [[[idx, con] for idx, con in enumerate(c_l)] for c_l in cons_list],
            'cons_list_rev': cons_list_rev,
            'parallel_data': option, 'table_list': table_list, 'rh_config': rh_config,
            'pareto_data': pareto,
            'task_inf': {
                'table_field': ['task_id', 'Advisor Type', 'Surrogate Type', 'max_runs',
                                'Time Limit Per Trial'],
                'table_data': [self.meta_data['task_id'], self.meta_data['advisor_type'],
                               self.meta_data['surrogate_type'], self.meta_data['max_iterations'],
                               self.meta_data['time_limit_per_trial']]
            },
            'importance_data': None,
            'pre_label_data': None,
            'grade_data': None,
            'cons_pre_label_data': None
        }
        return draw_data

    def generate_importance_data(self, method='shap'):
        try:
            if method != 'shap':  # todo: add other methods, such as fanova
                raise NotImplementedError('HTMLVisualizer only supports shap importance method currently!')

            importance_dict = self.history.get_importance(method=method, return_dict=True)
            if importance_dict is None or importance_dict == {}:
                return None

            objective_importance = importance_dict['objective_importance']
            constraint_importance = importance_dict['constraint_importance']
            X = self.history.get_config_array(transform='numerical')
            parameters = self.history.get_config_space().get_hyperparameter_names()

            objective_shap_values = np.asarray(importance_dict['objective_shap_values']).tolist()
            constraint_shap_values = np.asarray(importance_dict['constraint_shap_values']).tolist()

            importance = {
                'X': X.tolist(),
                'x': list(parameters),
                'data': dict(),
                'con_data': dict(),
                'obj_shap_value': objective_shap_values,
                'con_shap_value': constraint_shap_values,
            }

            for key, value in objective_importance.items():
                for i in range(len(value)):
                    y_name = 'opt-value-' + str(i + 1)
                    if y_name not in importance['data']:
                        importance['data'][y_name] = list()
                    importance['data'][y_name].append(value[i])

            for key, value in constraint_importance.items():
                for i in range(len(value)):
                    y_name = 'con-value-' + str(i + 1)
                    if y_name not in importance['con_data']:
                        importance['con_data'][y_name] = list()
                    importance['con_data'][y_name].append(value[i])

            return importance
        except Exception:
            logger.exception('Exception in generating importance data!')
            return None

    def generate_verify_surrogate_data(self):
        try:
            logger.info('Verify surrogate model...')

            from openbox.utils.config_space.util import convert_configurations_to_array
            # prepare object surrogate model data
            X_all = self.history.get_config_array(transform='scale')
            Y_all = self.history.get_objectives(transform='infeasible')

            if self.history.num_objectives == 1:  # todo: prf does not support copy. use build surrogate instead.
                models = [copy.deepcopy(self.surrogate_model)]
            else:
                models = copy.deepcopy(self.surrogate_model)
            pre_label_data, grade_data = self.verify_surrogate(X_all, Y_all, models)

            if self.history.num_constraints == 0:
                return pre_label_data, grade_data, None

            # prepare constraint surrogate model data
            cons_X_all = X_all
            cons_Y_all = self.history.get_constraints(transform='bilog')
            cons_models = copy.deepcopy(self.constraint_models)

            cons_pre_label_data, _ = self.verify_surrogate(cons_X_all, cons_Y_all, cons_models)

            return pre_label_data, grade_data, cons_pre_label_data
        except Exception:
            logger.exception('Exception in generating verify surrogate data!')
            return None, None, None

    def verify_surrogate(self, X_all, Y_all, models):
        assert models is not None

        # configuration number, obj/cons number
        N, num_objectives = Y_all.shape
        if X_all.shape[0] != N or N == 0:
            logger.error('Invalid data shape for verify_surrogate!')
            return None, None

        # 10-fold validation
        pre_perfs = [list() for i in range(num_objectives)]
        interval = math.ceil(N / 10)

        for i in range(num_objectives):
            for j in range(0, 10):
                X = np.concatenate((X_all[:j * interval, :], X_all[(j + 1) * interval:, :]), axis=0)
                Y = np.concatenate((Y_all[:j * interval, i], Y_all[(j + 1) * interval:, i]))
                tmp_model = copy.deepcopy(models[i])
                tmp_model.train(X, Y)

                test_X = X_all[j * interval:(j + 1) * interval, :]
                pre_mean, pre_var = tmp_model.predict(test_X)
                for tmp in pre_mean:
                    pre_perfs[i].append(tmp[0])

        ranks = [[0] * N for i in range(num_objectives)]
        pre_ranks = [[0] * N for i in range(num_objectives)]
        for i in range(num_objectives):
            tmp = np.argsort(Y_all[:, i]).astype(int)
            pre_tmp = np.argsort(pre_perfs[i]).astype(int)

            for j in range(N):
                ranks[i][tmp[j]] = j + 1
                pre_ranks[i][pre_tmp[j]] = j + 1

        min1 = float('inf')
        max1 = -float('inf')
        for i in range(num_objectives):
            min1 = min(min1, round(min(min(pre_perfs[i]), min(Y_all[:, i])), 3))
            max1 = max(max1, round(max(max(pre_perfs[i]), max(Y_all[:, i])), 3))
        min1 = min(min1, 0)

        pre_label_data = {
            'data': [list(zip(pre_perfs[i], Y_all[:, i])) for i in range(num_objectives)],
            'min': min1,
            'max': round(max1 * 1.1, 3)
        }
        grade_data = {
            'data': [list(zip(pre_ranks[i], ranks[i])) for i in range(num_objectives)],
            'min': 0,
            'max': self.meta_data['max_iterations']
        }

        return pre_label_data, grade_data

    def generate_html(self):
        try:
            # todo: isn’t compatible with PEP 302. should use importlib_resources to access data files.
            #   https://setuptools.pypa.io/en/latest/userguide/datafiles.html#accessing-data-files-at-runtime
            # todo: move static html files to assets/
            # static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'html/assets/static')
            static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../artifact/user_board/static')
            visual_static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'html/assets/static')
            template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'html/assets/visual_template.html')

            with open(template_path, 'r', encoding='utf-8') as f:
                html_text = f.read()

            link1_path = os.path.join(static_path, 'vendor/bootstrap/css/bootstrap.min.css')
            html_text = re.sub("<link rel=\"stylesheet\" href=\"../static/vendor/bootstrap/css/bootstrap.min.css\">",
                               "<link rel=\"stylesheet\" href=" + repr(link1_path) + ">", html_text)

            link2_path = os.path.join(static_path, 'css/style.default.css')
            html_text = re.sub(
                "<link rel=\"stylesheet\" href=\"../static/css/style.default.css\" id=\"theme-stylesheet\">",
                "<link rel=\"stylesheet\" href=" + repr(link2_path) + " id=\"theme-stylesheet\">", html_text)

            link3_path = os.path.join(static_path, 'css/custom.css')
            html_text = re.sub("<link rel=\"stylesheet\" href=\"../static/css/custom.css\">",
                               "<link rel=\"stylesheet\" href=" + repr(link3_path) + ">", html_text)

            html_text = re.sub("<script type=\"text/javascript\" src='json_path'></script>",
                               "<script type=\"text/javascript\" src=" + repr(self.json_path) + "></script>", html_text)

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

            script5_path = os.path.join(visual_static_path, 'js/echarts.min.js')
            html_text = re.sub("<script src=\"../static/js/echarts.min.js\"></script>",
                               "<script src=" + repr(script5_path) + "></script>", html_text)

            script6_path = os.path.join(static_path, 'js/common.js')
            html_text = re.sub("<script src=\"../static/js/common.js\"></script>",
                               "<script src=" + repr(script6_path) + "></script>", html_text)

            script7_path = os.path.join(visual_static_path, 'js/echarts-gl.min.js')
            html_text = re.sub("<script src=\"../static/js/echarts-gl.min.js\"></script>",
                               "<script src=" + repr(script7_path) + "></script>", html_text)

            with open(self.html_path, "w") as f:
                f.write(html_text)

            logger.info('Please open the html file to view visualization result: %s' % self.displayed_html_path)
        except Exception:
            logger.exception('Failed to generate html file!')

    def open_html(self):
        try:
            import webbrowser
            success = webbrowser.open(self.displayed_html_path)
            if not success:
                raise ValueError('webbrowser.open() returned False.')
        except Exception:
            logger.exception('Failed to open html file! Please open it manually: %s' % self.displayed_html_path)
