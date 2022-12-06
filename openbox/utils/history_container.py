# License: MIT

import sys
import time
import json
import collections
from typing import List, Union
import numpy as np
from ConfigSpace import CategoricalHyperparameter
from openbox.utils.constants import MAXINT, SUCCESS
from openbox.utils.config_space import Configuration, ConfigurationSpace
from openbox.utils.logging_utils import get_logger
from openbox.utils.multi_objective import Hypervolume, get_pareto_front
from openbox.utils.config_space.space_utils import get_config_from_dict, get_config_values, get_config_numerical_values
from openbox.core.base import Observation
from openbox.utils.transform import get_transform_function

Perf = collections.namedtuple(
    'perf', ['cost', 'time', 'status', 'additional_info'])


class HistoryContainer(object):
    def __init__(self, task_id, num_constraints=0, config_space=None):
        self.task_id = task_id
        self.config_space = config_space  # for show_importance
        self.data = collections.OrderedDict()  # only successful data
        self.config_counter = 0
        self.incumbent_value = MAXINT
        self.incumbents = list()
        self.logger = get_logger(self.__class__.__name__)

        self.num_objs = 1
        self.num_constraints = num_constraints
        self.configurations = list()  # all configurations (include successful and failed)
        self.perfs = list()  # all perfs
        self.constraint_perfs = list()  # all constraints
        self.trial_states = list()  # all trial states
        self.elapsed_times = list()  # all elapsed times

        self.update_times = list()  # record all update times

        self.successful_perfs = list()  # perfs of successful trials
        self.failed_index = list()
        self.transform_perf_index = list()

        self.global_start_time = time.time()
        self.scale_perc = 5
        self.perc = None
        self.min_y = None
        self.max_y = MAXINT

    def update_observation(self, observation: Observation):
        self.update_times.append(time.time() - self.global_start_time)

        config = observation.config
        objs = observation.objs
        constraints = observation.constraints
        trial_state = observation.trial_state
        elapsed_time = observation.elapsed_time

        self.configurations.append(config)
        if self.num_objs == 1:
            self.perfs.append(objs[0])
        else:
            self.perfs.append(objs)
        self.constraint_perfs.append(constraints)  # None if no constraint
        self.trial_states.append(trial_state)
        self.elapsed_times.append(elapsed_time)

        transform_perf = False
        failed = False
        if trial_state == SUCCESS and all(perf < MAXINT for perf in objs):
            if self.num_constraints > 0 and constraints is None:
                self.logger.error('Constraint is None in a SUCCESS trial!')
                failed = True
                transform_perf = True
            else:
                # If infeasible, transform perf to the largest found objective value
                feasible = True
                if self.num_constraints > 0 and any(c > 0 for c in constraints):
                    transform_perf = True
                    feasible = False

                if self.num_objs == 1:
                    self.successful_perfs.append(objs[0])
                    if feasible:
                        self.add(config, objs[0])
                    else:
                        self.add(config, MAXINT)
                else:
                    self.successful_perfs.append(objs)
                    if feasible:
                        self.add(config, objs)
                    else:
                        self.add(config, [MAXINT] * self.num_objs)

                self.perc = np.percentile(self.successful_perfs, self.scale_perc, axis=0)
                self.min_y = np.min(self.successful_perfs, axis=0).tolist()
                self.max_y = np.max(self.successful_perfs, axis=0).tolist()

        else:
            # failed trial
            failed = True
            transform_perf = True

        cur_idx = len(self.perfs) - 1
        if transform_perf:
            self.transform_perf_index.append(cur_idx)
        if failed:
            self.failed_index.append(cur_idx)

    def add(self, config: Configuration, perf: Perf):
        if config in self.data:
            self.logger.warning('Repeated configuration detected!')
            return

        self.data[config] = perf
        self.config_counter += 1

        if len(self.incumbents) > 0:
            if perf < self.incumbent_value:
                self.incumbents.clear()
            if perf <= self.incumbent_value:
                self.incumbents.append((config, perf))
                self.incumbent_value = perf
        else:
            self.incumbent_value = perf
            self.incumbents.append((config, perf))

    def get_transformed_perfs(self, transform=None):
        # set perf of failed trials to current max
        transformed_perfs = self.perfs.copy()
        for i in self.transform_perf_index:
            transformed_perfs[i] = self.max_y

        transformed_perfs = np.array(transformed_perfs, dtype=np.float64)
        transformed_perfs = get_transform_function(transform)(transformed_perfs)
        return transformed_perfs

    def get_transformed_constraint_perfs(self, transform='bilog'):
        if self.num_constraints == 0:
            return None

        transformed_constraint_perfs = self.constraint_perfs.copy()
        success_constraint_perfs = [c for c in transformed_constraint_perfs if c is not None]
        max_c = np.max(success_constraint_perfs, axis=0) if success_constraint_perfs else [1.0] * self.num_constraints
        for i in self.failed_index:
            transformed_constraint_perfs[i] = max_c

        transformed_constraint_perfs = np.array(transformed_constraint_perfs, dtype=np.float64)
        transformed_constraint_perfs = get_transform_function(transform)(transformed_constraint_perfs)
        return transformed_constraint_perfs

    def get_perf(self, config: Configuration):
        return self.data[config]

    def get_all_perfs(self):
        return list(self.data.values())

    def get_all_configs(self):
        return list(self.data.keys())

    def empty(self):
        return self.config_counter == 0

    def get_incumbents(self):
        return self.incumbents

    def get_str(self):
        from prettytable import PrettyTable
        incumbents = self.get_incumbents()
        if not incumbents:
            return 'No incumbents in history. Please run optimization process.'

        max_incumbents = 5
        if len(incumbents) > max_incumbents:
            self.logger.info(
                'Too many incumbents in history. Only show %d/%d of them.' % (max_incumbents, len(incumbents)))
            incumbents = incumbents[:max_incumbents]

        parameters = incumbents[0][0].configuration_space.get_hyperparameter_names()
        if len(incumbents) == 1:
            field_names = ["Parameters"] + ["Optimal Value"]
        else:
            field_names = ["Parameters"] + ["Optimal Value %d" % i for i in range(1, len(incumbents) + 1)]
        table = PrettyTable(field_names=field_names, float_format=".6", align="l")
        for param in parameters:
            row = [param] + [config.get_dictionary().get(param) for config, perf in incumbents]
            table.add_row(row)
        table.add_row(["Optimal Objective Value"] + [perf for config, perf in incumbents])
        table.add_row(["Num Configs"] + [len(self.configurations)] + [""] * (len(incumbents) - 1))

        # add hlines for the last 2 rows
        n_last_rows = 2
        raw_table = str(table)
        lines = raw_table.splitlines()
        hline = lines[2]
        for i in range(n_last_rows):
            lines.insert(-(i + 1) * 2, hline)
        render_table = "\n".join(lines)
        return render_table

    def __str__(self):
        return self.get_str()

    __repr__ = __str__

    def plot_convergence(
            self,
            true_minimum=None, name=None, clip_y=True,
            title="Convergence plot",
            xlabel="Iterations",
            ylabel="Min objective value",
            ax=None, alpha=0.3, yscale=None,
            color='C0', infeasible_color='C1',
            **kwargs):
        """Plot convergence trace.

        Parameters
        ----------
        true_minimum : float, optional
            True minimum value of the objective function.

        For other parameters, see `plot_convergence` in `openbox.visualization`.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The matplotlib axes.
        """
        from openbox.visualization import plot_convergence
        y = np.array(self.perfs, dtype=np.float64)  # do not transform infeasible trials
        cy = self.get_transformed_constraint_perfs(transform=None)
        ax = plot_convergence(y, cy, true_minimum, name, clip_y, title, xlabel, ylabel, ax, alpha, yscale,
                              color, infeasible_color, **kwargs)
        return ax

    def visualize_jupyter(self):
        try:
            import hiplot as hip
        except ModuleNotFoundError:
            if sys.version_info < (3, 6):
                raise ValueError("HiPlot requires Python 3.6 or newer. "
                                 "See https://facebookresearch.github.io/hiplot/getting_started.html")
            self.logger.error("Please run 'pip install hiplot'. "
                              "HiPlot requires Python 3.6 or newer.")
            raise

        visualize_data = []
        for config, perf in zip(self.configurations, self.perfs):
            config_perf = config.get_dictionary().copy()
            assert 'perf' not in config_perf.keys()
            config_perf['perf'] = perf
            visualize_data.append(config_perf)
        hip.Experiment.from_iterable(visualize_data).display()
        return

    def visualize_html(self, show_importance=False, verify_surrogate=False, optimizer=None, **kwargs):
        from openbox.visualization import build_visualizer, HTMLVisualizer
        # todo: user-friendly interface
        if optimizer is None:
            raise ValueError('Please provide optimizer for html visualization.')

        option = 'advanced' if (show_importance or verify_surrogate) else 'basic'
        visualizer = build_visualizer(option, optimizer=optimizer, **kwargs)  # type: HTMLVisualizer
        if visualizer.history_container is not self:
            visualizer.history_container = self
            visualizer.meta_data['task_id'] = self.task_id
        visualizer.visualize(show_importance=show_importance, verify_surrogate=verify_surrogate)
        return visualizer

    def get_importance(self, method='fanova', config_space=None, return_dict=False, return_allvalue=False):
        from prettytable import PrettyTable

        X = np.array([get_config_numerical_values(config) for config in self.configurations])
        Y = np.array(self.get_transformed_perfs(transform=None))
        if len(Y.shape) == 1:
            Y = np.reshape(Y, (len(Y), 1))
        if config_space is None:
            config_space = self.config_space
        if config_space is None:
            raise ValueError('Please provide config_space to show parameter importance!')
        keys = config_space.get_hyperparameter_names()
        importance_dict = {key: [] for key in keys}
        con_importance_dict = {key: [] for key in keys}

        if method == 'shap':
            import shap
            from lightgbm import LGBMRegressor

            for hp in config_space.get_hyperparameters():
                if isinstance(hp, CategoricalHyperparameter):
                    print("SHAP can not support categorical hyperparameters well. To analyze a space with categorical "
                          "hyperparameters, we recommend setting the method to fanova.")

            constraint_num = np.array(self.constraint_perfs)
            obj_shape_value = []
            con_shape_value = []

            for col_idx in range(self.num_constraints):
                # Fit a LightGBMRegressor with observations
                lgbr = LGBMRegressor(n_jobs=1)
                lgbr.fit(X, constraint_num[:, col_idx])
                explainer = shap.TreeExplainer(lgbr)
                shap_values = explainer.shap_values(X)
                if type(shap_values) == type(X):
                    con_shape_value.append(shap_values.tolist())
                else:
                    con_shape_value.append(shap_values)
                con_feature_importance = np.mean(np.abs(shap_values), axis=0)

                keys = [hp.name for hp in config_space.get_hyperparameters()]
                for i, hp_name in enumerate(keys):
                    con_importance_dict[hp_name].append(con_feature_importance[i])

            for col_idx in range(self.num_objs):
                # Fit a LightGBMRegressor with observations
                lgbr = LGBMRegressor(n_jobs=1)
                lgbr.fit(X, Y[:, col_idx])
                explainer = shap.TreeExplainer(lgbr)
                shap_values = explainer.shap_values(X)
                if type(shap_values) == type(X):
                    obj_shape_value.append(shap_values.tolist())
                else:
                    obj_shape_value.append(shap_values)
                feature_importance = np.mean(np.abs(shap_values), axis=0)

                keys = [hp.name for hp in config_space.get_hyperparameters()]
                for i, hp_name in enumerate(keys):
                    importance_dict[hp_name].append(feature_importance[i])
            
            if return_allvalue:
                return dict({'X': X.tolist(),
                    'obj_shap_value':obj_shape_value, 'importance_dict':importance_dict,
                    'con_shap_value':con_shape_value, 'con_importance_dict':con_importance_dict
                })

        elif method == 'fanova':
            try:
                import pyrfr.regression as reg
                import pyrfr.util
            except ModuleNotFoundError:
                self.logger.error(
                    'To use get_importance(), please install pyrfr: '
                    'https://open-box.readthedocs.io/en/latest/installation/install_pyrfr.html'
                )
                raise
            from openbox.utils.fanova import fANOVA

            if return_allvalue:  # todo
                raise NotImplementedError()

            for col_idx in range(self.num_objs):
                # create an instance of fanova with data for the random forest and the configSpace
                f = fANOVA(X=X, Y=Y[:, col_idx], config_space=config_space)

                # marginal for first parameter
                for key in keys:
                    p_list = (key,)
                    res = f.quantify_importance(p_list)
                    individual_importance = res[(key,)]['individual importance']
                    importance_dict[key].append(individual_importance)
        else:
            raise ValueError("Invalid method for feature importance. Valid choices: 'shap' or 'fanova'.")

        if return_dict:
            return importance_dict

        rows = []
        for param, values in importance_dict.items():
            rows.append([param, *values])
        if self.num_objs == 1:
            field_names = ["Parameter", "Importance"]
            rows.sort(key=lambda x: x[1], reverse=True)
        else:
            field_names = ["Parameter"] + ["Obj%d Importance" % i for i in range(1, self.num_objs + 1)]
        importance_table = PrettyTable(field_names=field_names, float_format=".6", align="l")
        importance_table.add_rows(rows)
        return importance_table

    def save_json(self, fn: str = "history_container.json"):
        """
        saves runhistory on disk

        Parameters
        ----------
        fn : str
            file name
        """

        data = []
        for idx, (config, perf, constraint_perf, trial_state, elapsed_time) in enumerate(zip(
                self.configurations, self.perfs, self.constraint_perfs, self.trial_states, self.elapsed_times,
        )):
            config_dict = config.get_dictionary()
            _perf = [float(p) for p in perf] if self.num_objs > 1 else float(perf)
            _constraint_perf = [float(c) for c in constraint_perf] if self.num_constraints > 0 else constraint_perf

            data_item = dict(
                index=idx,
                config=config_dict,
                perf=_perf,
                constraint_perf=_constraint_perf,
                trial_state=trial_state,
                elapsed_time=elapsed_time,
            )
            data.append(data_item)

        with open(fn, 'w') as fp:
            json.dump({'data': data}, fp, indent=2)
        self.logger.info('Save history to %s' % fn)

    def load_history_from_json(self, fn: str = "history_container.json", config_space: ConfigurationSpace = None):
        """Load history in json representation from disk.
        Parameters
        ----------
        fn : str
            file name to load from
        config_space : ConfigSpace
            instance of configuration space
        """

        if config_space is None:
            config_space = self.config_space
        if config_space is None:
            raise ValueError('Please provide config_space to load_history_from_json!')

        try:
            with open(fn, 'r') as fp:
                all_data = json.load(fp)
        except Exception as e:
            self.logger.warning(
                'Encountered exception %s while reading history from %s. '
                'Not adding any runs!', e, fn,
            )
            return

        for data_item in all_data['data']:
            config_dict = data_item['config']
            perf = data_item['perf']
            constraint_perf = data_item['constraint_perf']
            trial_state = data_item['trial_state']
            elapsed_time = data_item['elapsed_time']

            config = get_config_from_dict(config_space, config_dict)
            objs = perf if self.num_objs > 1 else [perf]

            observation = Observation(
                config=config, objs=objs, constraints=constraint_perf, trial_state=trial_state,
                elapsed_time=elapsed_time)
            self.update_observation(observation)

        self.logger.info('Load history from %s. len = %d.' % (fn, len(all_data['data'])))


class MOHistoryContainer(HistoryContainer):
    """
    Multi-Objective History Container
    """

    def __init__(self, task_id, num_objs, num_constraints=0, config_space=None, ref_point=None):
        super().__init__(task_id=task_id, num_constraints=num_constraints, config_space=config_space)
        self.pareto = collections.OrderedDict()
        self.num_objs = num_objs
        self.mo_incumbent_value = [MAXINT] * self.num_objs
        self.mo_incumbents = [list() for _ in range(self.num_objs)]
        self.ref_point = ref_point
        self.hv_data = list()

        self.max_y = [MAXINT] * self.num_objs

    def add(self, config: Configuration, perf: List[Perf]):
        assert self.num_objs == len(perf)

        if config in self.data:
            self.logger.warning('Repeated configuration detected!')
            return

        self.data[config] = perf
        self.config_counter += 1

        # update pareto
        remove_config = []
        for pareto_config, pareto_perf in self.pareto.items():  # todo efficient way?
            if all(pp <= p for pp, p in zip(pareto_perf, perf)):
                break
            elif all(p <= pp for pp, p in zip(pareto_perf, perf)):
                remove_config.append(pareto_config)
        else:
            self.pareto[config] = perf
            self.logger.info('Update pareto: config=%s, objs=%s.' % (str(config), str(perf)))

        for conf in remove_config:
            self.logger.info('Remove from pareto: config=%s, objs=%s.' % (str(conf), str(self.pareto[conf])))
            self.pareto.pop(conf)

        # update mo_incumbents
        for i in range(self.num_objs):
            if len(self.mo_incumbents[i]) > 0:
                if perf[i] < self.mo_incumbent_value[i]:
                    self.mo_incumbents[i].clear()
                if perf[i] <= self.mo_incumbent_value[i]:
                    self.mo_incumbents[i].append((config, perf[i], perf))
                    self.mo_incumbent_value[i] = perf[i]
            else:
                self.mo_incumbent_value[i] = perf[i]
                self.mo_incumbents[i].append((config, perf[i], perf))

        # Calculate current hypervolume if reference point is provided
        if self.ref_point is not None:
            pareto_front = self.get_pareto_front()
            if pareto_front:
                hv = Hypervolume(ref_point=self.ref_point).compute(pareto_front)
            else:
                hv = 0
            self.hv_data.append(hv)

    def get_incumbents(self):
        return self.get_pareto()

    def get_mo_incumbents(self):
        return self.mo_incumbents

    def get_mo_incumbent_value(self):
        return self.mo_incumbent_value

    def get_pareto(self):
        return list(self.pareto.items())

    def get_pareto_set(self):
        return list(self.pareto.keys())

    def get_pareto_front(self):
        return list(self.pareto.values())

    def compute_hypervolume(self, ref_point=None):
        if ref_point is None:
            ref_point = self.ref_point
        assert ref_point is not None
        pareto_front = self.get_pareto_front()
        if pareto_front:
            hv = Hypervolume(ref_point=ref_point).compute(pareto_front)
        else:
            hv = 0
        return hv

    def plot_convergence(self, *args, **kwargs):
        raise NotImplementedError('plot_convergence only supports single objective!')

    def visualize_jupyter(self, *args, **kwargs):
        raise NotImplementedError('visualize_jupyter only supports single objective!')


class MultiStartHistoryContainer(object):
    """
    History container for multistart algorithms.
    """

    def __init__(self, task_id, num_objs=1, num_constraints=0, config_space=None, ref_point=None):
        self.task_id = task_id
        self.num_objs = num_objs
        self.num_constraints = num_constraints
        self.history_containers = []
        self.config_space = config_space
        self.ref_point = ref_point
        self.current = None
        self.restart()

    def restart(self):
        if self.num_objs == 1:
            self.current = HistoryContainer(self.task_id, self.num_constraints, self.config_space)
        else:
            self.current = MOHistoryContainer(
                self.task_id, self.num_objs, self.num_constraints, self.config_space, self.ref_point)
        self.history_containers.append(self.current)

    def get_configs_for_all_restarts(self):
        all_configs = []
        for history_container in self.history_containers:
            all_configs.extend(list(history_container.data.keys()))
        return all_configs

    def get_incumbents_for_all_restarts(self):
        best_incumbents = []
        best_incumbent_value = float('inf')
        if self.num_objs == 1:
            for hc in self.history_containers:
                incumbents = hc.get_incumbents()
                incumbent_value = hc.incumbent_value
                if incumbent_value > best_incumbent_value:
                    continue
                elif incumbent_value < best_incumbent_value:
                    best_incumbent_value = incumbent_value
                best_incumbents.extend(incumbents)
            return best_incumbents
        else:
            return self.get_pareto_front()

    def get_pareto_front(self):
        assert self.num_objs > 1
        Y = np.vstack([hc.get_pareto_front() for hc in self.history_containers])
        return get_pareto_front(Y).tolist()

    def update_observation(self, observation: Observation):
        return self.current.update_observation(observation)

    def add(self, config: Configuration, perf: Perf):
        self.current.add(config, perf)

    @property
    def configurations(self):
        return self.current.configurations

    @property
    def perfs(self):
        return self.current.perfs

    @property
    def constraint_perfs(self):
        return self.current.constraint_perfs

    @property
    def trial_states(self):
        return self.current.trial_states

    @property
    def successful_perfs(self):
        return self.current.successful_perfs

    def get_transformed_perfs(self, *args, **kwargs):
        return self.current.get_transformed_perfs(*args, **kwargs)

    def get_transformed_constraint_perfs(self, *args, **kwargs):
        return self.current.get_transformed_constraint_perfs(*args, **kwargs)

    def get_perf(self, config: Configuration):
        for history_container in self.history_containers:
            if config in history_container.data:
                return self.data[config]
        raise KeyError

    def get_all_configs(self):
        return self.current.get_all_configs()

    def empty(self):
        return self.current.config_counter == 0

    def get_incumbents(self):
        if self.num_objs == 1:
            return self.current.incumbents
        else:
            return self.current.get_pareto()

    def get_mo_incumbents(self):
        assert self.num_objs > 1
        return self.current.mo_incumbents

    def get_mo_incumbent_value(self):
        assert self.num_objs > 1
        return self.current.mo_incumbent_value

    def get_pareto(self):
        assert self.num_objs > 1
        return self.current.get_pareto()

    def get_pareto_set(self):
        assert self.num_objs > 1
        return self.current.get_pareto_set()

    def compute_hypervolume(self, ref_point=None):
        assert self.num_objs > 1
        return self.current.compute_hypervolume(ref_point)

    def save_json(self, fn: str = "history_container.json"):
        """
        saves runhistory on disk

        Parameters
        ----------
        fn : str
            file name
        """
        self.current.save_json(fn)

    def load_history_from_json(self, fn: str = "history_container.json", config_space: ConfigurationSpace = None):
        """Load history in json representation from disk.
        Parameters
        ----------
        fn : str
            file name to load from
        config_space : ConfigSpace
            instance of configuration space
        """
        self.current.load_history_from_json(fn, config_space)
