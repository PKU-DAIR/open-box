# License: MIT

import time
import json
import copy
from datetime import datetime
import collections
from typing import List, Union, Optional
from functools import partial
import numpy as np
from ConfigSpace import CategoricalHyperparameter, OrdinalHyperparameter
from openbox import logger
from openbox.utils.constants import SUCCESS
from openbox.utils.config_space import Configuration, ConfigurationSpace
from openbox.utils.multi_objective import Hypervolume, get_pareto_front
from openbox.utils.config_space.space_utils import get_config_from_dict, get_config_values, get_config_numerical_values
from openbox.utils.transform import get_transform_function
from openbox.utils.config_space.util import convert_configurations_to_array
from openbox.utils.util_funcs import deprecate_kwarg, transform_to_1d_list


class Observation(object):
    @deprecate_kwarg('objs', 'objectives', 'a future version')
    def __init__(
            self,
            config: Configuration,
            objectives: Union[List[float], np.ndarray],
            constraints: Optional[Union[List[float], np.ndarray]] = None,
            trial_state: Optional['State'] = SUCCESS,
            elapsed_time: Optional[float] = None,
            extra_info: Optional[dict] = None,
    ):
        self.config = config
        self.objectives = objectives
        self.constraints = constraints
        self.trial_state = trial_state
        self.elapsed_time = elapsed_time
        self.create_time = datetime.now()
        if extra_info is None:
            extra_info = dict()
        assert isinstance(extra_info, dict)
        self.extra_info = extra_info

        self.objectives = transform_to_1d_list(self.objectives, hint='objectives')
        if self.constraints is not None:
            self.constraints = transform_to_1d_list(self.constraints, hint='constraints')

    def __str__(self):
        items = [f'config={self.config}', f'objectives={self.objectives}']
        if self.constraints is not None:
            items.append(f'constraints={self.constraints}')
        items.append(f'trial_state={self.trial_state}')
        if self.elapsed_time is not None:
            items.append(f'elapsed_time={self.elapsed_time}')
        items.append(f'create_time={self.create_time}')
        if self.extra_info:
            items.append(f'extra_info={self.extra_info}')
        return f'Observation({", ".join(items)})'

    __repr__ = __str__

    def to_dict(self):
        d = {
            'config': self.config.get_dictionary(),
            'objectives': self.objectives,
            'constraints': self.constraints,
            'trial_state': self.trial_state,
            'elapsed_time': self.elapsed_time,
            'create_time': self.create_time,
            'extra_info': self.extra_info,
        }
        for k, v in d.items():
            d[k] = copy.deepcopy(v)

        if isinstance(d['create_time'], datetime):
            d['create_time'] = d['create_time'].isoformat()

        return d

    @classmethod
    def from_dict(cls, d: dict, config_space: ConfigurationSpace):
        config = d['config']
        if isinstance(config, dict):
            assert config_space is not None, 'config_space must be provided if config is a dict'
            d['config'] = get_config_from_dict(config_space, config)

        create_time = d.pop('create_time', None)

        observation = cls(**d)

        if isinstance(create_time, str):
            observation.create_time = datetime.fromisoformat(create_time)
        return observation

    def __eq__(self, other):
        if not isinstance(other, Observation):
            return False
        return self.to_dict() == other.to_dict()


class HistoryContainer(object):
    def __init__(self, task_id, num_constraints=0, config_space=None):
        self.task_id = task_id
        self.config_space = config_space  # for show_importance
        self.data = collections.OrderedDict()  # only successful data
        self.config_counter = 0
        self.incumbent_value = np.inf
        self.incumbents = list()

        self.num_objectives = 1
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
        self.max_y = np.inf

    def update_observation(self, observation: Observation):
        self.update_times.append(time.time() - self.global_start_time)

        config = observation.config
        objectives = observation.objectives
        constraints = observation.constraints
        trial_state = observation.trial_state
        elapsed_time = observation.elapsed_time

        self.configurations.append(config)
        if self.num_objectives == 1:
            self.perfs.append(objectives[0])
        else:
            self.perfs.append(objectives)
        self.constraint_perfs.append(constraints)  # None if no constraint
        self.trial_states.append(trial_state)
        self.elapsed_times.append(elapsed_time)

        transform_perf = False
        failed = False
        if trial_state == SUCCESS and all(perf < np.inf for perf in objectives):
            if self.num_constraints > 0 and constraints is None:
                logger.error('Constraint is None in a SUCCESS trial!')
                failed = True
                transform_perf = True
            else:
                # If infeasible, transform perf to the largest found objective value
                feasible = True
                if self.num_constraints > 0 and any(c > 0 for c in constraints):
                    transform_perf = True
                    feasible = False

                if self.num_objectives == 1:
                    self.successful_perfs.append(objectives[0])
                    if feasible:
                        self.add(config, objectives[0])
                    else:
                        self.add(config, np.inf)
                else:
                    self.successful_perfs.append(objectives)
                    if feasible:
                        self.add(config, objectives)
                    else:
                        self.add(config, [np.inf] * self.num_objectives)

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

    def add(self, config: Configuration, perf):
        if config in self.data:
            logger.warning('Repeated configuration detected!')
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

    def get_config_space(self):
        if hasattr(self, 'config_space') and self.config_space is not None:
            return self.config_space
        elif len(self) > 0:
            config_space = self.configurations[0].configuration_space
            return config_space
        else:
            raise ValueError('No config_space is set and no observation is recorded!')

    def get_converted_config_array(self):
        """
        Get the converted configuration array. Typically used for surrogate model training.

        Integer and float hyperparameters are scaled to [0, 1].
        Categorical and ordinal hyperparameters are transformed to index.

        Returns
        -------
        X: np.ndarray
            Converted configuration array. Shape: (n_configs, n_dims)
        """
        X = convert_configurations_to_array(self.configurations)
        return X

    def get_numerical_config_array(self):
        """
        Get the numerical configuration array.

        Integer and float hyperparameters are not scaled.
        Categorical and ordinal hyperparameters are transformed to index.

        Returns
        -------
        X: np.ndarray
            Numerical configuration array. Shape: (n_configs, n_dims)
        """
        X = np.array([get_config_numerical_values(config) for config in self.configurations])
        return X

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
        return len(self) == 0

    def get_incumbents(self):
        return self.incumbents

    def get_str(self):
        from prettytable import PrettyTable
        incumbents = self.get_incumbents()
        if not incumbents:
            return 'No incumbents in history. Please run optimization process.'

        max_incumbents = 5
        if len(incumbents) > max_incumbents:
            logger.info(
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

    def __len__(self):
        return len(self.configurations)

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

    def visualize_hiplot(self, html_file: Optional[str] = None, **kwargs):
        """
        Visualize the history using HiPlot in Jupyter Notebook.

        HiPlot documentation: https://facebookresearch.github.io/hiplot/

        Parameters
        ----------
        html_file: str, optional
            If None, the visualization will be shown in the juptyer notebook.
            If specified, the visualization will be saved to the html file.
        kwargs: dict
            Other keyword arguments passed to `hiplot.Experiment.display` or `hiplot.Experiment.to_html`.

        Returns
        -------
        exp: hiplot.Experiment
            The hiplot experiment object.
        """
        from openbox.visualization import visualize_hiplot

        configs = self.configurations
        y = np.array(self.perfs)
        cy = np.array(self.constraint_perfs) if self.num_constraints > 0 else None
        exp = visualize_hiplot(configs=configs, y=y, cy=cy, html_file=html_file, **kwargs)
        return exp

    def visualize_html(self, open_html=True, show_importance=False, verify_surrogate=False, optimizer=None, **kwargs):
        from openbox.visualization import build_visualizer, HTMLVisualizer
        # todo: user-friendly interface
        if optimizer is None:
            raise ValueError('Please provide optimizer for html visualization.')

        option = 'advanced' if (show_importance or verify_surrogate) else 'basic'
        visualizer = build_visualizer(option, optimizer=optimizer, **kwargs)  # type: HTMLVisualizer
        if visualizer.history_container is not self:
            visualizer.history_container = self
            visualizer.meta_data['task_id'] = self.task_id
        visualizer.visualize(open_html=open_html, show_importance=show_importance, verify_surrogate=verify_surrogate)
        return visualizer

    def get_importance(self, method='fanova', return_dict=False):
        """
        Feature importance analysis.

        Parameters
        ----------
        method : ['fanova', 'shap']
            Method to compute feature importance.
        return_dict : bool
            Whether to return a dict of feature importance.

        Returns
        -------
        importance : dict or prettytable.PrettyTable
            If return_dict=True, return a dict of feature importance.
            If return_dict=False, return a prettytable.PrettyTable of feature importance.
                The table can be printed directly.
        """
        from prettytable import PrettyTable
        from openbox.utils.feature_importance import get_fanova_importance, get_shap_importance

        if len(self) == 0:
            logger.error('No observations in history! Please run optimization process.')
            return dict() if return_dict else None

        config_space = self.configurations[0].configuration_space
        parameters = list(config_space.get_hyperparameter_names())

        if method == 'fanova':
            importance_func = partial(get_fanova_importance, config_space=config_space)
        elif method == 'shap':
            # todo: try different hyperparameter in lgb
            importance_func = get_shap_importance
            if any([isinstance(hp, (CategoricalHyperparameter, OrdinalHyperparameter))
                    for hp in config_space.get_hyperparameters()]):
                logger.warning("SHAP can not support categorical/ordinal hyperparameters well. "
                               "To analyze a space with categorical/ordinal hyperparameters, "
                               "we recommend setting the method to fanova.")
        else:
            raise ValueError("Invalid method for feature importance: %s" % method)

        X = self.get_numerical_config_array()
        Y = self.get_transformed_perfs(transform=None)
        Y = Y.reshape(-1, self.num_objectives)
        cY = self.get_transformed_constraint_perfs(transform='bilog')

        importance_dict = {
            'objective_importance': {param: [] for param in parameters},
            'constraint_importance': {param: [] for param in parameters},
        }
        if method == 'shap':
            importance_dict['objective_shap_values'] = []
            importance_dict['constraint_shap_values'] = []

        for i in range(self.num_objectives):
            feature_importance = importance_func(X, Y[:, i])
            if method == 'shap':
                feature_importance, shap_values = feature_importance
                importance_dict['objective_shap_values'].append(shap_values)

            for param, importance in zip(parameters, feature_importance):
                importance_dict['objective_importance'][param].append(importance)

        for i in range(self.num_constraints):
            feature_importance = importance_func(X, cY[:, i])
            if method == 'shap':
                feature_importance, shap_values = feature_importance
                importance_dict['constraint_shap_values'].append(shap_values)

            for param, importance in zip(parameters, feature_importance):
                importance_dict['constraint_importance'][param].append(importance)

        if return_dict:
            return importance_dict

        # plot table
        rows = []
        for param in parameters:
            row = [param, *importance_dict['objective_importance'][param],
                   *importance_dict['constraint_importance'][param]]
            rows.append(row)
        if self.num_objectives == 1 and self.num_constraints == 0:
            field_names = ["Parameter", "Importance"]
            rows.sort(key=lambda x: x[1], reverse=True)
        else:
            field_names = ["Parameter"] + ["Obj%d Importance" % i for i in range(1, self.num_objectives + 1)] + \
                          ["Cons%d Importance" % i for i in range(1, self.num_constraints + 1)]
        importance_table = PrettyTable(field_names=field_names, float_format=".6", align="l")
        importance_table.add_rows(rows)
        return importance_table  # the table can be printed directly

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
            _perf = [float(p) for p in perf] if self.num_objectives > 1 else float(perf)
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
        logger.info('Save history to %s' % fn)

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
            logger.warning(
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
            objectives = perf if self.num_objectives > 1 else [perf]

            observation = Observation(
                config=config, objectives=objectives, constraints=constraint_perf, trial_state=trial_state,
                elapsed_time=elapsed_time)
            self.update_observation(observation)

        logger.info('Load history from %s. len = %d.' % (fn, len(all_data['data'])))


class MOHistoryContainer(HistoryContainer):
    """
    Multi-Objective History Container
    """

    def __init__(self, task_id, num_objectives, num_constraints=0, config_space=None, ref_point=None):
        super().__init__(task_id=task_id, num_constraints=num_constraints, config_space=config_space)
        self.pareto = collections.OrderedDict()
        self.num_objectives = num_objectives
        self.mo_incumbent_value = [np.inf] * self.num_objectives
        self.mo_incumbents = [list() for _ in range(self.num_objectives)]
        self.ref_point = ref_point
        self.hv_data = list()

        self.max_y = [np.inf] * self.num_objectives

    def add(self, config: Configuration, perf):
        assert self.num_objectives == len(perf)

        if config in self.data:
            logger.warning('Repeated configuration detected!')
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
            logger.info('Update pareto: config=%s, objectives=%s.' % (str(config), str(perf)))

        for conf in remove_config:
            logger.info('Remove from pareto: config=%s, objectives=%s.' % (str(conf), str(self.pareto[conf])))
            self.pareto.pop(conf)

        # update mo_incumbents
        for i in range(self.num_objectives):
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


class MultiStartHistoryContainer(object):
    """
    History container for multistart algorithms.
    """

    def __init__(self, task_id, num_objectives=1, num_constraints=0, config_space=None, ref_point=None):
        self.task_id = task_id
        self.num_objectives = num_objectives
        self.num_constraints = num_constraints
        self.history_containers = []
        self.config_space = config_space
        self.ref_point = ref_point
        self.current = None
        self.restart()

    def restart(self):
        if self.num_objectives == 1:
            self.current = HistoryContainer(self.task_id, self.num_constraints, self.config_space)
        else:
            self.current = MOHistoryContainer(
                self.task_id, self.num_objectives, self.num_constraints, self.config_space, self.ref_point)
        self.history_containers.append(self.current)

    def get_configs_for_all_restarts(self):
        all_configs = []
        for history_container in self.history_containers:
            all_configs.extend(list(history_container.data.keys()))
        return all_configs

    def get_incumbents_for_all_restarts(self):
        best_incumbents = []
        best_incumbent_value = float('inf')
        if self.num_objectives == 1:
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
        assert self.num_objectives > 1
        Y = np.vstack([hc.get_pareto_front() for hc in self.history_containers])
        return get_pareto_front(Y).tolist()

    def update_observation(self, observation: Observation):
        return self.current.update_observation(observation)

    def add(self, config: Configuration, perf):
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
        if self.num_objectives == 1:
            return self.current.incumbents
        else:
            return self.current.get_pareto()

    def get_mo_incumbents(self):
        assert self.num_objectives > 1
        return self.current.mo_incumbents

    def get_mo_incumbent_value(self):
        assert self.num_objectives > 1
        return self.current.mo_incumbent_value

    def get_pareto(self):
        assert self.num_objectives > 1
        return self.current.get_pareto()

    def get_pareto_set(self):
        assert self.num_objectives > 1
        return self.current.get_pareto_set()

    def compute_hypervolume(self, ref_point=None):
        assert self.num_objectives > 1
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
