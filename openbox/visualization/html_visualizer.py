import os
import re
import time
import json
import math
import copy
from typing import List, Union
import numpy as np
from openbox.utils.history_container import HistoryContainer
from openbox.visualization.base_visualizer import BaseVisualizer
from openbox.surrogate.base.base_model import AbstractModel


class HTMLVisualizer(BaseVisualizer):
    _default_advanced_analysis_options = dict(
        importance_update_interval=5,
    )

    def __init__(
            self,
            logging_dir: str,
            history_container: HistoryContainer,
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
        assert isinstance(logging_dir, str) and logging_dir
        task_id = history_container.task_id
        self.output_dir = os.path.join(logging_dir, "history/%s/" % task_id)
        os.makedirs(self.output_dir, exist_ok=True)

        self.advanced_analysis = advanced_analysis
        if advanced_analysis_options is None:
            advanced_analysis_options = dict()
        self.advanced_analysis_options = self._default_advanced_analysis_options.copy()
        self.advanced_analysis_options.update(advanced_analysis_options)
        self._cache_advanced_data = None

        self.history_container = history_container
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

    def setup(self):
        self.timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.generate_html()  # todo: check file conflict

    def update(self, update_importance=None, verify_surrogate=None):
        iter_id = len(self.history_container.configurations)
        max_iter = self.meta_data['max_iterations'] or np.inf
        if update_importance is None:
            update_interval = self.advanced_analysis_options['importance_update_interval']
            update_importance = iter_id and ((iter_id % update_interval == 0) or (iter_id >= max_iter))
        if verify_surrogate is None:
            verify_surrogate = iter_id >= max_iter
        self.save_visualization_data(update_importance=update_importance, verify_surrogate=verify_surrogate)

    def visualize(self, show_importance=False, verify_surrogate=False):
        self.setup()
        self.update(update_importance=show_importance, verify_surrogate=verify_surrogate)

    def generate_html(self):
        pass

    def save_visualization_data(self, update_importance=False, verify_surrogate=False):
        pass


def save_visualization_data(history_container, logging_dir, timestamp, advisor_type, surrogate_type, max_iterations, time_limit_per_trial, verify_sur, surrogate_model, constraint_models):
    task_id = history_container.task_id

    vis_log_dir = os.path.join(logging_dir, "history/%s/" % task_id)
    # with open(os.path.join(vis_log_dir, "%s_%s.json" % (task_id, timestamp)), 'w') as fp:
    #     json.dump({'data': data}, fp, indent=2)

    processed_data = process_data(history_container, advisor_type, surrogate_type, max_iterations, time_limit_per_trial, verify_sur, surrogate_model, constraint_models)
    with open(os.path.join(vis_log_dir, 'visual_%s_%s.json' % (task_id, timestamp)), 'w') as fp:
        fp.write('var info=')
        json.dump({'data': processed_data}, fp, indent=2)
        fp.write(';')


def process_data(his_con, advisor_type, surrogate_type, max_iterations, time_limit_per_trial, verify_sur, surrogate_model, constraint_models):
    # Config Table data
    table_list = []
    # all the config list
    rh_config = {}
    # Parallel Data
    option = {'data': [list() for i in range(his_con.num_objs)], 'schema': [], 'visualMap': {}}
    # all the performance
    perf_list = [list() for i in range(his_con.num_objs)]
    # all the constraints, A[i][j]: 第i个约束中第j个配置对应的值
    cons_list = [list() for i in range(his_con.num_constraints)]
    # A[i][j]: 第i个个配置对应的第j个约束值
    cons_list_rev = list()

    for idx in range(len(his_con.perfs)):
        results = [round(tmp, 4) for tmp in his_con.perfs[idx]]
        constraints = None
        if his_con.num_constraints > 0:
            constraints = [round(tmp, 4) for tmp in his_con.constraint_perfs[idx]]
            cons_list_rev.append(constraints)

        config_dic = his_con.configurations[idx].get_dictionary()
        config_str = str(config_dic)
        if len(config_str) > 35:
            config_str = config_str[1:35]
        else:
            config_str = config_str[1:-1]

        table_list.append(
            [idx+1, results, constraints, config_str, his_con.trial_states[idx], round(his_con.elapsed_times[idx], 3)])

        rh_config[str(idx+1)] = config_dic

        config_values = []
        for parameter in config_dic.keys():
            config_values.append(config_dic[parameter])

        for i in range(his_con.num_objs):
            option['data'][i].append(config_values + [results[i]])

        for i in range(his_con.num_objs):
            perf_list[i].append(results[i])

        for i in range(his_con.num_constraints):
            cons_list[i].append(constraints[i])

    if len(his_con.perfs) > 0:
        option['schema'] = list(his_con.configurations[0].get_dictionary().keys()) + ['perf']
        mi = float('inf')
        ma = -float('inf')
        for i in range(his_con.num_objs):
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
    # ok: 符合约束，并且在最下沿；no：不符合约束；other：符合约束，不再最下沿
    line_data = [{'ok': [], 'no': [], 'other': []} for i in range(his_con.num_objs)]

    for i in range(his_con.num_objs):
        min_value = float("inf")
        for idx, perf in enumerate(perf_list[i]):
            if his_con.num_constraints > 0 and np.any([cons_list_rev[idx][k] > 0 for k in range(his_con.num_constraints)]):
                line_data[i]['no'].append([idx, perf])
                continue
            if perf <= min_value:
                min_value = perf
                line_data[i]['ok'].append([idx, perf])
            else:
                line_data[i]['other'].append([idx, perf])
        line_data[i]['ok'].append([len(option['data'][i]), min_value])

    # Pareto data
    pareto = dict({})
    if his_con.num_objs > 1:
        pareto["ref_point"] = his_con.ref_point
        pareto["hv"] = [[idx, round(v, 3)] for idx, v in enumerate(his_con.hv_data)]
        pareto["pareto_point"] = list(his_con.pareto.values())
        pareto["all_points"] = his_con.perfs

    # Importance data
    history_dict = his_con.get_importance(method='shap', return_allvalue=True)
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
        'num_objs': his_con.num_objs, 'num_constraints': his_con.num_constraints,
        'line_data': line_data,
        'cons_line_data': [[[idx, con] for idx, con in enumerate(c_l)] for c_l in cons_list],
        'cons_list_rev': cons_list_rev,
        'parallel_data': option, 'table_list': table_list, 'rh_config': rh_config,
        'importance_data': importance,
        'pareto_data': pareto,
        'task_inf': {
            'table_field': ['task_id', 'Advisor Type', 'Surrogate Type', 'max_runs',
                            'Time Limit Per Trial'],
            'table_data': [his_con.task_id, advisor_type, surrogate_type, max_iterations,
                           time_limit_per_trial]
        },
        'pre_label_data': None,
        'grade_data': None,
        'cons_pre_label_data': None
    }

    if verify_sur:
        draw_data['pre_label_data'], draw_data['grade_data'] = verify_surrogate(his_con, surrogate_model, max_iterations)
        if his_con.num_constraints > 0:
            draw_data['cons_pre_label_data'] = cons_verify_surrogate(his_con, constraint_models)

    return draw_data


def verify_surrogate(his_con, surrogate_model, max_iterations):
    # only get successful observations
    confs = [his_con.configurations[i] for i in range(len(his_con.configurations)) if i not in his_con.failed_index]
    if his_con.num_objs == 1:
        perfs = [his_con.successful_perfs]
    else:
        perfs = [[per[i] for per in his_con.successful_perfs] for i in range(his_con.num_objs)]

    N = len(perfs[0])
    if len(confs) != N or N == 0:
        print("No equal!")
        return None, None

    from openbox.utils.config_space.util import convert_configurations_to_array
    X_all = convert_configurations_to_array(confs)
    Y_all = [np.array(perfs[i], dtype=np.float64) for i in range(his_con.num_objs)]
    if his_con.num_objs == 1:
        models = [copy.deepcopy(surrogate_model)]
    else:
        models = copy.deepcopy(surrogate_model)

    # 10-fold validation
    pre_perfs = [list() for i in range(his_con.num_objs)]
    interval = math.ceil(N / 10)

    for i in range(his_con.num_objs):
        for j in range(0, 10):
            X = np.concatenate((X_all[:j*interval, :], X_all[(j+1)*interval:, :]), axis=0)
            Y = np.concatenate((Y_all[i][:j*interval], Y_all[i][(j+1)*interval:]))
            # 如果是多目标，那么这就会是一个模型列表
            tmp_model = copy.deepcopy(models[i])
            tmp_model.train(X, Y)

            test_X = X_all[j*interval:(j+1)*interval, :]
            pre_mean, pre_var = tmp_model.predict(test_X)
            # 这里多目标可能要改
            for tmp in pre_mean:
                pre_perfs[i].append(tmp[0])

    ranks = [[0] * N for i in range(his_con.num_objs)]
    pre_ranks = [[0] * N for i in range(his_con.num_objs)]
    for i in range(his_con.num_objs):
        tmp = np.argsort(perfs[i]).astype(int)
        pre_tmp = np.argsort(pre_perfs[i]).astype(int)

        for j in range(N):
            ranks[i][tmp[j]] = j + 1
            pre_ranks[i][pre_tmp[j]] = j + 1

    min1 = float('inf')
    max1 = -float('inf')
    for i in range(his_con.num_objs):
        min1 = min(min1, round(min(min(pre_perfs[i]), min(perfs[i])), 3))
        max1 = max(max1, round(max(max(pre_perfs[i]), max(perfs[i])), 3))
    min1 = min(min1, 0)

    return {
               'data': [list(zip(pre_perfs[i], perfs[i])) for i in range(his_con.num_objs)],
               'min': min1,
               'max': round(max1 * 1.1, 3)
           }, \
           {
               'data': [list(zip(pre_ranks[i], ranks[i])) for i in range(his_con.num_objs)],
               'min': 0,
               'max': max_iterations
           }


def cons_verify_surrogate(his_con, constraint_models):
    confs = his_con.configurations
    # 每个实例的cons都是一个列表
    cons_perfs = [[tmp[i] for tmp in his_con.constraint_perfs] for i in range(his_con.num_constraints)]
    print(cons_perfs)

    N = len(cons_perfs[0])
    if len(confs) != N or N == 0:
        print("No equal!")
        return None, None

    from openbox.utils.config_space.util import convert_configurations_to_array
    X_all = convert_configurations_to_array(confs)
    Y_all = [np.array(cons_perfs[i], dtype=np.float64) for i in range(his_con.num_constraints)]
    models = copy.deepcopy(constraint_models)

    # leave one out validation
    pre_perfs = [list() for i in range(his_con.num_constraints)]
    interval = math.ceil(N / 10)

    for i in range(his_con.num_constraints):
        for j in range(0, 10):
            X = np.concatenate((X_all[:j*interval, :], X_all[(j+1)*interval:, :]), axis=0)
            Y = np.concatenate((Y_all[i][:j*interval], Y_all[i][(j+1)*interval:]))
            # 如果是多目标，那么这就会是一个模型列表
            tmp_model = copy.deepcopy(models[i])
            tmp_model.train(X, Y)

            test_X = X_all[j*interval:(j+1)*interval, :]
            pre_mean, pre_var = tmp_model.predict(test_X)
            # 这里多目标可能要改
            for tmp in pre_mean:
                pre_perfs[i].append(tmp[0])

    print(pre_perfs)
    min1 = float('inf')
    max1 = -float('inf')
    for i in range(his_con.num_objs):
        min1 = min(min1, round(min(min(pre_perfs[i]), min(cons_perfs[i])), 3))
        max1 = max(max1, round(max(max(pre_perfs[i]), max(cons_perfs[i])), 3))

    min1 = min(min1, 0)
    return {
        'data': [list(zip(pre_perfs[i], cons_perfs[i])) for i in range(his_con.num_constraints)],
        'min': min1,
        'max': round(max1 * 1.1, 3)
    }


def generate_html(logging_dir, task_id, timestamp):
    vis_log_dir = os.path.abspath(os.path.join(logging_dir, "history/%s/" % task_id))
    os.makedirs(vis_log_dir, exist_ok=True)

    static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets/static')
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets/visual_template.html')

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

    visual_json_path = os.path.join(vis_log_dir, 'visual_' + "%s_%s.json" % (task_id, timestamp))
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

    html_path = os.path.join(vis_log_dir, "%s_%s.html" % (task_id, timestamp))
    with open(html_path, "w+") as f:
        f.write(html_text)
