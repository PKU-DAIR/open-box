import os
import re
from datetime import datetime
import json
import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import KFold
from openbox import logger
from openbox.utils.history import History
from openbox.visualization.base_visualizer import BaseVisualizer
from openbox.core.base import build_surrogate


class HTMLVisualizer(BaseVisualizer):
    _default_advanced_analysis_options = dict(
        importance_update_interval=5,
        importance_method='shap'
    )
    _task_info_keys = [
        'task_id',
        'advisor_type', 'max_iterations', 'time_limit_per_trial',
        'surrogate_type', 'constraint_surrogate_type', 'transfer_learning_history'
    ]

    def __init__(
            self,
            logging_dir: str,
            history: History,
            task_info: dict,
            auto_open_html: bool = False,
            advanced_analysis: bool = False,
            advanced_analysis_options: dict = None,
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

        if task_info is None:
            task_info = dict()
        self.task_info = {
            'task_id': task_id,
        }
        self.task_info.update(task_info)
        for k in self._task_info_keys:
            if k not in self.task_info:
                self.task_info[k] = None

        self.timestamp = None
        self.html_path = None
        self.displayed_html_path = None
        self.json_path = None

        if self.advanced_analysis:
            self.check_dependency()

    def setup(self, open_html=None):
        self.timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        task_id = self.task_info['task_id']
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
        max_iter = self.task_info['max_iterations'] or np.inf
        if update_importance is None:
            if not self.advanced_analysis:
                update_importance = False
            else:
                update_interval = self.advanced_analysis_options['importance_update_interval']
                update_importance = iter_id and ((iter_id % update_interval == 0) or (iter_id >= max_iter))
        if verify_surrogate is None:
            verify_surrogate = False if not self.advanced_analysis \
                else (iter_id >= max_iter and self.task_info['surrogate_type'])
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
                importance = self.generate_importance_data(method=self.advanced_analysis_options['importance_method'])
                self._cache_advanced_data['importance'] = importance
            draw_data['importance_data'] = importance
            # verify surrogate data
            if verify_surrogate:
                pred_label_data, grade_data, cons_pred_label_data = self.generate_verify_surrogate_data()
                draw_data['pred_label_data'] = pred_label_data
                draw_data['grade_data'] = grade_data
                draw_data['cons_pred_label_data'] = cons_pred_label_data

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
        # # all the config list
        # rh_config = {}
        # Parallel Data
        option = {'data': [], 'schema': [], 'visualMap': {}}
        # all the performance
        perf_list = [list() for i in range(self.history.num_objectives)]
        # all the constraints, A[i][j]: value of constraint i, configuration j
        cons_list = [list() for i in range(self.history.num_constraints)]
        # A[i][j]: value of configuration i, constraint j
        cons_list_rev = list()

        # todo: check if has invalid value
        for idx, obs in enumerate(self.history.observations):
            objectives = [round(v, 6) for v in obs.objectives]
            constraints = None
            if self.history.num_constraints > 0:
                constraints = [round(v, 6) for v in obs.constraints]
                cons_list_rev.append(constraints)

            config_dic = obs.config.get_dictionary()
            config_str = str(config_dic)
            if len(config_str) > 35:
                config_str = config_str[1:35]
            else:
                config_str = config_str[1:-1]

            table_list.append(
                [idx + 1, objectives, constraints, config_dic, config_str, obs.trial_state,
                 round(obs.elapsed_time, 3)])

            config_values = list(config_dic.values())

            op = [idx + 1] + config_values + objectives
            if constraints:
                op = op + constraints
            option['data'].append(op)

            for i in range(self.history.num_objectives):
                perf_list[i].append(objectives[i])

            for i in range(self.history.num_constraints):
                cons_list[i].append(constraints[i])

        if len(self.history) > 0:
            parameters = self.history.get_config_space().get_hyperparameter_names()

            option['schema'] = ['Uid'] + list(parameters) \
                               + ['Objs ' + str(i + 1) for i in range(self.history.num_objectives)] \
                               + ['Cons ' + str(i + 1) for i in range(self.history.num_constraints)]

            option['visualMap']['min'] = 1
            option['visualMap']['max'] = len(self.history)

            option['visualMap']['dimension'] = 0
        else:
            option['visualMap']['min'] = 0
            option['visualMap']['max'] = 100
            option['visualMap']['dimension'] = 0

        # Line Data
        line_data = [{'best': [], 'infeasible': [], 'feasible': []} for i in range(self.history.num_objectives)]

        for i in range(self.history.num_objectives):
            min_value = float("inf")
            for idx, perf in enumerate(perf_list[i]):
                if self.history.num_constraints > 0 and np.any(
                        [cons_list_rev[idx][k] > 0 for k in range(self.history.num_constraints)]):
                    line_data[i]['infeasible'].append([idx + 1, perf])
                    continue
                if perf <= min_value:
                    min_value = perf
                    line_data[i]['best'].append([idx + 1, perf])
                else:
                    line_data[i]['feasible'].append([idx + 1, perf])
            # line_data[i]['best'].append([len(option['data'][i]), min_value])

        # Pareto data
        # todo: if has invalid value?
        y = self.history.get_objectives(transform='none', warn_invalid_value=False)
        success_mask = self.history.get_success_mask()
        feasible_mask = self.history.get_feasible_mask()

        pareto = dict()
        if self.history.num_objectives > 1:
            pareto["ref_point"] = self.history.ref_point
            if pareto["ref_point"] is None:
                pareto["hv"] = None
                logger.warning("Please provide ref_point for visualizer to draw hypervolume chart!\n")
            else:
                hypervolumes = self.history.compute_hypervolume(data_range='all')
                pareto["hv"] = [[idx + 1, round(v, 3)] for idx, v in enumerate(hypervolumes)]

            pareto["pareto_point"] = self.history.get_pareto_front(lexsort=True).tolist()
            pareto["pareto_point_feasible"] = y[feasible_mask & success_mask].tolist()
            pareto["pareto_point_infeasible"] = y[(~feasible_mask) & success_mask].tolist()
            pareto["all_points"] = self.history.get_objectives(transform='none', warn_invalid_value=True).tolist()

        draw_data = {
            'num_objectives': self.history.num_objectives, 'num_constraints': self.history.num_constraints,
            'advance': self.advanced_analysis,
            'line_data': line_data,
            'cons_line_data': [[[idx + 1, con] for idx, con in enumerate(c_l)] for c_l in cons_list],
            'cons_list_rev': cons_list_rev,
            'parallel_data': option, 'table_list': table_list,
            'pareto_data': pareto,
            'task_inf': {
                'table_field': ['Task Id', 'Advisor Type', 'Surrogate Type', 'Current Run', 'Max Runs',
                                'Time Limit Per Trial'],
                'table_data': [self.task_info['task_id'], self.task_info['advisor_type'],
                               self.task_info['surrogate_type'], len(self.history), self.task_info['max_iterations'],
                               self.task_info['time_limit_per_trial']]
            },
            'importance_data': None,
            'pred_label_data': None,
            'grade_data': None,
            'cons_pred_label_data': None
        }
        return draw_data

    def generate_importance_data(self, method):
        try:
            importance_dict = self.history.get_importance(method=method, return_dict=True)
            if importance_dict is None or importance_dict == {}:
                return None

            objective_importance = importance_dict['objective_importance']
            constraint_importance = importance_dict['constraint_importance']
            X = self.history.get_config_array(transform='numerical')
            parameters = self.history.get_config_space().get_hyperparameter_names()

            importance = {
                'X': X.tolist(),
                'x': list(parameters),
                'method': method,
                'data': dict(),
                'con_data': dict()
            }

            if method == 'shap':
                objective_shap_values = np.asarray(importance_dict['objective_shap_values']).tolist()
                constraint_shap_values = np.asarray(importance_dict['constraint_shap_values']).tolist()
                importance['obj_shap_value'] = objective_shap_values
                importance['con_shap_value'] = constraint_shap_values

            for key, value in objective_importance.items():
                for i in range(len(value)):
                    y_name = 'obj ' + str(i + 1)
                    if y_name not in importance['data']:
                        importance['data'][y_name] = list()
                    importance['data'][y_name].append(value[i])

            for key, value in constraint_importance.items():
                for i in range(len(value)):
                    y_name = 'cons ' + str(i + 1)
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
            X = self.history.get_config_array(transform='scale')
            Y = self.history.get_objectives(transform='infeasible')

            surrogate_type = self.task_info['surrogate_type']
            if surrogate_type is None:
                raise ValueError('Please set surrogate_type in task_info!')
            models = [build_surrogate(func_str=surrogate_type,
                                      config_space=self.history.get_config_space(),
                                      rng=np.random.RandomState(1),
                                      transfer_learning_history=self.task_info['transfer_learning_history'])
                      for _ in range(self.history.num_objectives)]

            pred_label_data, grade_data = self.verify_surrogate(X, Y, models)

            if self.history.num_constraints == 0:
                return pred_label_data, grade_data, None

            # prepare constraint surrogate model data
            cons_X = X
            cons_Y = self.history.get_constraints(transform='bilog')
            constraint_surrogate_type = self.task_info['constraint_surrogate_type']
            if constraint_surrogate_type is None:
                raise ValueError('Please set constraint_surrogate_type in task_info!')
            cons_models = [build_surrogate(func_str=constraint_surrogate_type,
                                           config_space=self.history.get_config_space(),
                                           rng=np.random.RandomState(1))
                           for _ in range(self.history.num_constraints)]

            cons_pred_label_data, _ = self.verify_surrogate(cons_X, cons_Y, cons_models)

            return pred_label_data, grade_data, cons_pred_label_data
        except Exception:
            logger.exception('Exception in generating verify surrogate data!')
            return None, None, None

    def verify_surrogate(self, X, Y, models):
        assert models is not None

        # configuration number, obj/cons number
        N, num_objectives = Y.shape
        if X.shape[0] != N or N == 0:
            logger.error('Invalid data shape for verify_surrogate!')
            return None, None

        # cross validation
        pred_Y = np.zeros((N, num_objectives))
        ranks = np.zeros((N, num_objectives)).astype(int)
        pred_ranks = np.zeros((N, num_objectives)).astype(int)

        k = min(N, 5)
        kf = KFold(n_splits=k, shuffle=True, random_state=1024)
        for i in range(num_objectives):
            for train_index, test_index in kf.split(X):
                X_train, Y_train = X[train_index, :], Y[train_index, i]
                X_test = X[test_index, :]
                tmp_model = models[i]
                tmp_model.train(X_train, Y_train)

                pred_mean, _ = tmp_model.predict(X_test)
                pred_Y[test_index, i:i + 1] = pred_mean

            rank = rankdata(Y[:, i], method='min')
            pred_rank = rankdata(pred_Y[:, i], method='min')

            ranks[:, i] = rank
            pred_ranks[:, i] = pred_rank

        min_array = np.min(np.concatenate([Y, pred_Y], axis=0), axis=0)
        max_array = np.max(np.concatenate([Y, pred_Y], axis=0), axis=0)
        interval = (max_array - min_array) * 0.05
        min_array = np.round(min_array - interval, 3)
        max_array = np.round(max_array + interval, 3)

        pred_label_data = {
            'data': [list(zip(pred_Y[:, i].tolist(), Y[:, i].tolist())) for i in range(num_objectives)],
            'min_array': min_array.tolist(),
            'max_array': max_array.tolist(),
        }
        grade_data = {
            'data': [list(zip(pred_ranks[:, i].tolist(), ranks[:, i].tolist())) for i in range(num_objectives)],
            'min': 0,
            'max': len(self.history),
        }

        return pred_label_data, grade_data

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

            relative_json_path = os.path.basename(self.json_path)
            html_text = re.sub("<script type=\"text/javascript\" src='json_path'></script>",
                               "<script type=\"text/javascript\" src=" + repr(relative_json_path) + "></script>",
                               html_text)

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
