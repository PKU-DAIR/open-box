# License: MIT

import os
import copy
import json
from datetime import datetime
from functools import partial
from typing import List, Tuple, Union, Optional
import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, CategoricalHyperparameter, OrdinalHyperparameter
from openbox import logger
from openbox.utils.constants import SUCCESS
from openbox.utils.config_space.util import convert_configurations_to_array
from openbox.utils.config_space.space_utils import get_config_from_dict, get_config_values, get_config_numerical_values
from openbox.utils.transform import get_transform_function
from openbox.utils.multi_objective import get_pareto_front, Hypervolume
from openbox.utils.util_funcs import transform_to_1d_list, deprecate_kwarg


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
        data = {
            'config': self.config.get_dictionary(),
            'objectives': self.objectives,
            'constraints': self.constraints,
            'trial_state': self.trial_state,
            'elapsed_time': self.elapsed_time,
            'create_time': self.create_time.isoformat(),
            'extra_info': self.extra_info,
        }
        data = copy.deepcopy(data)
        return data

    @classmethod
    def from_dict(cls, data: dict, config_space: ConfigurationSpace):
        config = data['config']
        if isinstance(config, dict):
            assert config_space is not None, 'config_space must be provided if config is a dict'
            data['config'] = get_config_from_dict(config_space, config)
        else:
            assert isinstance(config, Configuration), 'config must be a dict or Configuration'

        create_time = data.pop('create_time', None)

        observation = cls(**data)

        if isinstance(create_time, str):
            observation.create_time = datetime.fromisoformat(create_time)
        else:
            logger.warning(f'Unable to parse create_time ({create_time}) from dict.')
        return observation

    def __eq__(self, other):
        if not isinstance(other, Observation):
            return False
        return self.to_dict() == other.to_dict()


class History(object):
    """
    A history object stores the observations of the optimization process.

    Parameters
    ----------
    task_id: str
        Task id.
    num_objectives: int, default=1
        Number of objectives.
    num_constraints: int, default=0
        Number of constraints.
    config_space: ConfigurationSpace, optional
        Configuration space.
    ref_point: list or np.ndarray, optional
        Reference point for multi-objective hypervolume calculation.
    meta_info: dict, optional
        Meta information.
    """
    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    def __init__(
            self,
            task_id: str = 'OpenBox',
            num_objectives: int = 1,
            num_constraints: int = 0,
            config_space: Optional[ConfigurationSpace] = None,
            ref_point: Optional[Union[List[float], np.ndarray]] = None,
            meta_info: Optional[dict] = None,
    ):
        self.task_id = task_id
        self.num_objectives = num_objectives
        self.num_constraints = num_constraints
        self.config_space = config_space
        if meta_info is None:
            meta_info = dict()
        assert isinstance(meta_info, dict)
        self.meta_info = meta_info

        self.observations = []
        self.global_start_time = datetime.now()

        # multi-objective
        self._ref_point = None
        self.ref_point = ref_point  # type: Optional[List[float]]

    def __len__(self):
        return len(self.observations)

    def empty(self):
        return len(self) == 0

    @property
    def configurations(self) -> List[Configuration]:
        return [obs.config for obs in self.observations]

    @property
    def objectives(self) -> List[List[float]]:
        return [obs.objectives for obs in self.observations]

    @property
    def constraints(self) -> List[Optional[List[float]]]:
        return [obs.constraints for obs in self.observations]

    # alias
    configs = configurations
    objs = objectives
    constrs = cons = constraints

    @property
    def trial_states(self) -> List['State']:
        return [obs.trial_state for obs in self.observations]

    @property
    def elapsed_times(self) -> List[Optional[float]]:
        return [obs.elapsed_time for obs in self.observations]

    @property
    def create_times(self) -> List[datetime]:
        return [obs.create_time for obs in self.observations]

    @property
    def extra_infos(self) -> List[dict]:
        return [obs.extra_info for obs in self.observations]

    @property
    def ref_point(self) -> Optional[List[float]]:
        return self._ref_point

    @ref_point.setter
    def ref_point(self, ref_point: Optional[Union[List[float], np.ndarray]]):
        if ref_point is not None:
            assert self.num_objectives > 1, 'Reference point is only used for multi-objective optimization!'
        self._ref_point = self.check_ref_point(ref_point)  # type: Optional[List[float]]

    def check_ref_point(self, ref_point: Optional[Union[List[float], np.ndarray]]) -> Optional[List[float]]:
        """check and standardize the reference point"""
        if ref_point is not None:
            assert self.num_objectives > 1, 'Reference point is only used for multi-objective optimization!'
            ref_point = transform_to_1d_list(ref_point, hint='ref_point')
            assert len(ref_point) == self.num_objectives, 'Length of ref_point must be equal to num_objectives'
        return ref_point

    @staticmethod
    def _has_invalid_value(x: np.ndarray) -> bool:
        """Check if x has invalid value (nan, inf, -inf)."""
        x = np.asarray(x, dtype=np.float64)
        return np.any(np.isnan(x)) or np.any(np.isinf(x))

    def is_valid_observation(self, obs: Observation, raise_error=True):
        """Check if the observation is valid. If raise_error=True, raise ValueError if invalid."""
        try:
            if not isinstance(obs, Observation):
                raise ValueError(f'observation must be an instance of Observation, got {type(obs)}')
            # here we do not check config_space, retain the flexibility of dynamic config space
            if not isinstance(obs.config, Configuration):
                raise ValueError(f'config must be an instance of Configuration, got {type(obs.config)}')
            if len(obs.objectives) != self.num_objectives:
                raise ValueError(f'num objectives must be {self.num_objectives}, got {len(obs.objectives)}')
            if obs.trial_state == SUCCESS and self._has_invalid_value(obs.objectives):
                raise ValueError(f'invalid values (inf, nan) are not allowed in objectives in a SUCCESS trial, '
                                 f'got {obs.objectives}')
            if self.num_constraints > 0 and obs.trial_state == SUCCESS:
                if obs.constraints is None:  # constraints might be None in an unsuccessful trial
                    raise ValueError(f'constraints is None in a SUCCESS trial!')
                if self._has_invalid_value(obs.constraints):
                    raise ValueError(f'invalid values (inf, nan) are not allowed in constraints in a SUCCESS trial, '
                                     f'got {obs.constraints}')
            if obs.constraints is not None and len(obs.constraints) != self.num_constraints:
                raise ValueError(f'num constraints must be {self.num_constraints}, got {len(obs.constraints)}')
            if not isinstance(obs.extra_info, dict):
                raise ValueError(f'extra_info must be a dict, got {type(obs.extra_info)}')
        except Exception:
            logger.exception(f'Invalid observation: {obs}')
            if raise_error:
                raise
            return False
        return True

    def update_observation(self, observation: Observation) -> None:
        """Update the observation to the history."""
        self.is_valid_observation(observation, raise_error=True)
        if observation.config in self.configurations:
            logger.warning('Duplicate configuration detected!')
        self.observations.append(observation)
        logger.debug(f'Observation updated in history: {observation}')

    def update_observations(self, observations: List[Observation]) -> None:
        """Update a list of observations to the history."""
        # check all observations first
        for observation in observations:
            self.is_valid_observation(observation, raise_error=True)
        for observation in observations:
            self.update_observation(observation)
        logger.info(f'{len(observations)} observations updated in history.')

    def get_config_space(self) -> Optional[ConfigurationSpace]:
        """
        Get configuration space.
        - If config_space is not provided in __init__, it will be inferred from the first observation.
        - Then, if no observation is present, return None.

        Returns
        -------
        config_space: Optional[ConfigurationSpace]
            Configuration space.
        """
        if self.config_space is not None:
            return self.config_space
        elif len(self) > 0:
            config_space = self.configurations[0].configuration_space
            return config_space
        else:
            logger.warning('Failed to get config_space because it is not set in History '
                           'and no observation is recorded. Return None.')
            return None

    def get_config_array(self, transform: str = 'scale') -> np.ndarray:
        """
        Get the configuration array
        - Integer and float hyperparameters are transformed according to the `transform` parameter.
        - Categorical and ordinal hyperparameters are transformed to index.

        Parameters
        ----------
        transform: ['scale', 'numerical'], default='scale'
            Transform method for integer and float hyperparameters.
            - 'scale': Scale the integer and float hyperparameters to [0, 1].
                       Typically used for surrogate model training.
            - 'numerical': Keep the integer and float hyperparameters as they are.

        Returns
        -------
        config_array: np.ndarray
            Configuration array. Shape: (n_configs, n_dims)
        """
        if transform == 'scale':
            return convert_configurations_to_array(self.configurations)
        elif transform == 'numerical':
            return np.array([get_config_numerical_values(config) for config in self.configurations])
        else:
            raise ValueError(f'Unknown transform method: {transform}')

    def get_config_dicts(self) -> List[dict]:
        """
        Get a list of configuration dictionaries.

        Returns
        -------
        config_dicts: List[dict]
            A list of configuration dictionaries.
        """
        return [copy.deepcopy(config.get_dictionary()) for config in self.configurations]

    @staticmethod
    def _get_min_max_values(X: np.ndarray, axis: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get min and max value of x along the given axis.
        - np.nan, np.inf and -np.inf are ignored.
        - If no value is valid, return 0 for that dimension.

        Parameters
        ----------
        X: np.ndarray
            Input array.
        axis: int
            Axis along which to operate.

        Returns
        -------
        min_X: np.ndarray
            Min value of X along the given axis.
        max_X: np.ndarray
            Max value of X along the given axis.
        """
        X = np.asarray(X, dtype=np.float64)
        # replace inf with nan
        X[np.isinf(X)] = np.nan
        # get min and max value of X along the given axis
        min_X = np.nanmin(X, axis=axis)
        max_X = np.nanmax(X, axis=axis)
        # replace nan with 0
        min_X[np.isnan(min_X)] = 0.0
        max_X[np.isnan(max_X)] = 0.0
        return min_X, max_X

    def _get_transformed_values(self, attr: str, transform: str, warn_invalid_value: bool = True) -> np.ndarray:
        """
        Get objectives / constraints as np.ndarray and perform transformation (e.g., failed handling).

        Parameters
        ----------
        attr: ['objectives', 'constraints']
            Attribute to get.
        transform: str
            Transform method. Multiple methods can be combined with comma (,).
            - 'none': Return the original objectives. Invalid values (nan, inf) are kept.
            If not 'none', failed trials are always replaced with max observed values.
                Note that for successful trials, invalid values are not allowed.
            Additional options:
            - 'failed': Replace failed trials with max observed values.
                        Set to this option if you don't want to use other transformations.
            - 'infeasible': Replace infeasible trials with max observed values.
                            Do not set this option if attr='constraints'!
            - Others: Use transform function in `openbox.utils.transform.get_transform_function`.
            todo: currently, normalization is handled in surrogate model.
              If you add normalization here, take care of other functions like get_incumbents
        warn_invalid_value: bool, default=True
            Whether to log warning if invalid values are found and no transformation is applied (transform='none').

        Returns
        -------
        values: np.ndarray
            (Transformed) Values. Shape: (n_configs, num_objectives) or (n_configs, num_constraints)
        """
        if attr == 'objectives':
            values = self.objectives
        elif attr == 'constraints':
            values = self.constraints
        else:
            raise ValueError(f'Unknown attribute: {attr}. Must be "objectives" or "constraints".')

        assert isinstance(transform, str)
        transform = transform.lower()

        values = np.asarray(values, dtype=np.float64)
        if transform == 'none':
            if warn_invalid_value and self._has_invalid_value(values):
                logger.warning(f'{attr} contains invalid values (nan or inf) and is returned as is.')
            return values

        # handle multiple transforms
        transform = set(map(str.strip, transform.split(',')))
        if '' in transform:
            transform.remove('')

        # If transform != 'none', trials that are failed or have inf values
        # are always replaced with max observed values
        if 'failed' in transform:
            transform.remove('failed')
        success_mask = self.get_success_mask()
        values[~success_mask] = np.full(values.shape[1], np.nan)  # do not compute max/min for failed trials
        min_values, max_values = self._get_min_max_values(values, axis=0)
        values[~success_mask] = max_values

        if 'infeasible' in transform:
            transform.remove('infeasible')
            if attr == 'constraints':
                raise ValueError('Cannot use "infeasible" transform for constraints!')
            feasible_mask = self.get_feasible_mask()
            values[~feasible_mask] = max_values

        for tf in transform:
            values = get_transform_function(tf)(values)

        values = np.asarray(values, dtype=np.float64)
        return values

    def get_objectives(self, transform: str = 'infeasible', warn_invalid_value: bool = True) -> np.ndarray:
        """
        Get objectives as np.ndarray and perform transformation (e.g., failed/infeasible handling).

        Parameters
        ----------
        transform: str, default='infeasible'
            Transform method. Multiple methods can be combined with comma (,).
            - 'none': Return the original objectives. Invalid values (nan, inf) are kept.
            If not 'none', failed trials are always replaced with max observed values.
                Note that for successful trials, invalid values are not allowed.
            Additional options:
            - 'failed': Replace failed trials with max observed values.
                        Set to this option if you don't want to use other transformations.
            - 'infeasible': Replace infeasible trials with max observed values.
            - Others: Use transform function in `openbox.utils.transform.get_transform_function`.
        warn_invalid_value: bool, default=True
            Whether to log warning if invalid values are found and no transformation is applied (transform='none').

        Returns
        -------
        objectives: np.ndarray
            (Transformed) Objectives. Shape: (n_configs, num_objectives)
        """
        objectives = self._get_transformed_values(
            attr='objectives', transform=transform, warn_invalid_value=warn_invalid_value)
        return objectives

    def get_constraints(self, transform: str = 'bilog', warn_invalid_value: bool = True) -> Optional[np.ndarray]:
        """
        Get constraints as np.ndarray and perform transformation (e.g., failed handling).

        Parameters
        ----------
        transform: str, default='bilog'
            Transform method. Multiple methods can be combined with comma (,).
            - 'none': Return the original objectives. Invalid values (nan, inf) are kept.
            If not 'none', failed trials are always replaced with max observed values.
                Note that for successful trials, invalid values are not allowed.
            Additional options:
            - 'failed': Replace failed trials with max observed values.
                        Set to this option if you don't want to use other transformations.
            - Others: Use transform function in `openbox.utils.transform.get_transform_function`.
                      E.g., 'bilog' for bilog transformation.
        warn_invalid_value: bool, default=True
            Whether to log warning if invalid values are found and no transformation is applied (transform='none').

        Returns
        -------
        constraints: Optional[np.ndarray]
            (Transformed) Constraints. Shape: (n_configs, num_constraints)
            If no constraints, return None.
        """
        if self.num_constraints == 0:
            return None
        constraints = self._get_transformed_values(
            attr='constraints', transform=transform, warn_invalid_value=warn_invalid_value)
        return constraints

    def get_success_mask(self) -> np.ndarray:
        """
        Get the mask of successful trials.

        Returns
        -------
        success_mask: np.ndarray
            Success mask. Shape: (n_configs, )
        """
        success_mask = np.asarray([trial_state == SUCCESS for trial_state in self.trial_states], dtype=bool)
        return success_mask

    def get_success_count(self) -> int:
        """
        Get the number of successful trials.

        Returns
        -------
        success_count: int
            Number of successful trials.
        """
        cnt = np.sum(self.get_success_mask())
        return int(cnt.item())

    def get_feasible_mask(self, exclude_failed: bool = True) -> np.ndarray:
        """
        Get the mask of feasible trials.

        Parameters
        ----------
        exclude_failed: bool, default=True
            If True, failed trials are always considered infeasible,
                even if they may have feasible constraints in very rare cases.

        Returns
        -------
        feasible_mask: np.ndarray
            Feasible mask. Shape: (n_configs, )
        """
        if self.num_constraints == 0:
            feasible_mask = np.ones(len(self), dtype=bool)
        else:
            constraints = self.get_constraints(transform='none', warn_invalid_value=False)  # keeping nan and inf is ok
            feasible_mask = np.all(constraints <= 0, axis=-1)  # constraints <= 0 means feasible
        if exclude_failed:
            feasible_mask &= self.get_success_mask()  # failed trials are not feasible
        return feasible_mask

    def get_feasible_count(self, exclude_failed: bool = True) -> int:
        """
        Get the number of feasible trials.

        Parameters
        ----------
        exclude_failed: bool, default=True
            If True, failed trials are always considered infeasible,
                even if they may have feasible constraints in very rare cases.

        Returns
        -------
        feasible_count: int
            Number of feasible trials.
        """
        cnt = np.sum(self.get_feasible_mask(exclude_failed=exclude_failed))
        return int(cnt.item())

    def get_incumbents(self) -> List[Observation]:
        """
        Get incumbent observations.
        - If multiple incumbents have the same objective values, all of them will be returned.
        - Only feasible incumbents will be returned.
        - Only used for single-objective optimization (num_objectives=1).

        Returns
        -------
        incumbents: List[Observation]
            Incumbent observations.
        """
        if self.num_objectives > 1:
            raise ValueError('get_incumbents() is used for single-objective optimization! Use get_pareto() instead.')

        feasible_mask = self.get_feasible_mask(exclude_failed=True)
        if not np.any(feasible_mask):
            logger.warning('No feasible incumbent observations returned!')
            return []

        objectives = self.get_objectives(transform='none', warn_invalid_value=False)
        # no invalid value in SUCCESS trials
        incumbent_value = np.min(objectives[feasible_mask])  # num_objectives=1
        incumbent_mask = (objectives == incumbent_value).reshape(-1) & feasible_mask

        incumbents = [self.observations[i] for i in np.where(incumbent_mask)[0]]
        return incumbents

    def get_incumbent_value(self) -> float:
        """
        Get incumbent value.
        - Best objective value of feasible trials.
        - Only used for single-objective optimization.

        Returns
        -------
        incumbent_value: float
            Incumbent value.
        """
        if self.num_objectives > 1:
            raise ValueError('get_incumbent_value() is used for single-objective optimization! '
                             'Use get_pareto_front() instead.')

        feasible_mask = self.get_feasible_mask(exclude_failed=True)
        if not np.any(feasible_mask):
            logger.warning('No feasible observations! Return np.inf as incumbent value.')
            return np.inf

        objectives = self.get_objectives(transform='none', warn_invalid_value=False)
        # no invalid value in SUCCESS trials
        incumbent_value = np.min(objectives[feasible_mask])  # num_objectives=1
        return incumbent_value

    def get_incumbent_configs(self) -> List[Configuration]:
        """
        Get incumbent configurations.
        - If multiple incumbents have the same objective values, all of them will be returned.
        - Only feasible configurations will be returned.
        - Only used for single-objective optimization.

        Returns
        -------
        incumbent_configs: List[Configuration]
            Incumbent configurations.
        """
        if self.num_objectives > 1:
            raise ValueError('get_incumbent_configs() is used for single-objective optimization! '
                             'Use get_pareto_set() instead.')
        incumbents = self.get_incumbents()
        incumbent_configs = [obs.config for obs in incumbents]
        return incumbent_configs

    def get_mo_incumbent_values(self) -> np.ndarray:
        """
        Get incumbent value of each objective for multi-objectives.
        - Only used for multi-objective optimization.

        Returns
        -------
        mo_incumbent_values: np.ndarray
            Incumbent values of multi-objectives.
        """
        assert self.num_objectives > 1

        feasible_mask = self.get_feasible_mask(exclude_failed=True)
        if not np.any(feasible_mask):
            logger.warning('No feasible observations! Return np.inf(s) as incumbent values.')
            return np.full(self.num_objectives, np.inf)

        objectives = self.get_objectives(transform='none', warn_invalid_value=False)
        # no invalid value in SUCCESS trials
        mo_incumbent_values = np.min(objectives[feasible_mask], axis=0)
        return mo_incumbent_values

    def get_pareto(self) -> List[Observation]:
        """
        Get pareto observations.
        - Only feasible observations will be returned.
        - Only used for multi-objective optimization.

        Returns
        -------
        pareto: List[Observation]
            Pareto observations.
        """
        assert self.num_objectives > 1

        feasible_mask = self.get_feasible_mask(exclude_failed=True)
        if not np.any(feasible_mask):
            logger.warning('No feasible observations! Return empty pareto.')
            return []

        objectives = self.get_objectives(transform='none', warn_invalid_value=False)
        # no invalid value in SUCCESS trials
        pareto_idx = get_pareto_front(objectives[feasible_mask], return_index=True)  # idx of feasible trials
        pareto = [self.observations[i] for i in np.where(feasible_mask)[0][pareto_idx]]
        return pareto

    def get_pareto_front(self, lexsort: bool = True) -> np.ndarray:
        """
        Get pareto front.
        - Only feasible trials will be considered.
        - Only used for multi-objective optimization.

        Parameters
        ----------
        lexsort: bool
            Whether to sort the pareto front by lexicographical order.

        Returns
        -------
        pareto_front: np.ndarray
            Pareto front. Shape: (num_pareto, num_objectives)
        """
        assert self.num_objectives > 1

        feasible_mask = self.get_feasible_mask(exclude_failed=True)
        if not np.any(feasible_mask):
            logger.warning('No feasible observations! Return empty pareto front.')
            return np.empty((0, self.num_objectives), dtype=np.float64)

        objectives = self.get_objectives(transform='none', warn_invalid_value=False)
        # no invalid value in SUCCESS trials
        pareto_front = get_pareto_front(objectives[feasible_mask], lexsort=lexsort)
        return pareto_front

    def get_pareto_set(self) -> List[Configuration]:
        """
        Get pareto set (configurations).
        - Only feasible configurations will be returned.
        - Only used for multi-objective optimization.

        Returns
        -------
        pareto_set: List[Configuration]
            Pareto set (configurations).
        """
        assert self.num_objectives > 1

        pareto = self.get_pareto()
        pareto_set = [obs.config for obs in pareto]
        return pareto_set

    def compute_hypervolume(
            self,
            ref_point: Optional[List[float]] = None,
            data_range: str = 'last',
    ) -> Union[float, List[float]]:
        """
        Compute hypervolume of the pareto front.
        - Only used for multi-objective optimization.

        Parameters
        ----------
        ref_point: Optional[List[float]]
            Reference point for hypervolume calculation. If None, use self.ref_point.
        data_range: ['last', 'all']
            If 'last', only compute hypervolume of the last pareto front.
            If 'all', compute hypervolumes during the whole optimization process.

        Returns
        -------
        hypervolume: Union[float, List[float]]
            If data_range='last', return a float value.
            If data_range='all', return a list of float values.
        """
        assert self.num_objectives > 1
        ref_point = self.check_ref_point(ref_point)
        ref_point = ref_point if ref_point is not None else self.ref_point
        assert ref_point is not None, 'ref_point must be provided!'

        if data_range == 'last':
            pareto_front = self.get_pareto_front(lexsort=False)  # type: np.ndarray  # empty array is allowed
            hv = Hypervolume(ref_point=ref_point).compute(pareto_front)
            return hv
        elif data_range == 'all':
            logger.info('Computing all hypervolumes...')
            feasible_mask = self.get_feasible_mask(exclude_failed=True)
            objectives = self.get_objectives(transform='none', warn_invalid_value=False)
            HV = Hypervolume(ref_point=ref_point)
            hv_list = []
            for i in range(len(self)):
                mask = feasible_mask[:i + 1]
                objs = objectives[:i + 1]
                pareto_front = get_pareto_front(objs[mask], lexsort=False)
                hv = HV.compute(pareto_front)
                hv_list.append(hv)
            if len(self) == 0:
                logger.warning('No observations! Return empty hypervolume list.')
            return hv_list
        else:
            raise ValueError(f'Invalid data_range: {data_range}')

    def get_str(self, max_candidates: int = 5) -> str:
        from prettytable import PrettyTable

        if self.empty():
            return 'No observation in History. Please run optimization.'

        candidates = self.get_incumbents() if self.num_objectives == 1 else self.get_pareto()  # type: List[Observation]
        n_candidates = len(candidates)  # record the number of candidates before truncation
        if len(candidates) > max_candidates:
            hint = 'incumbents in history' if self.num_objectives == 1 else 'points on Pareto front'
            logger.info(f'Too many {hint}. Only show {max_candidates}/{n_candidates} of them.')
            candidates = candidates[:max_candidates]

        parameters = self.get_config_space().get_hyperparameter_names()
        if len(candidates) == 1:
            field_names = ["Parameters"] + ["Optimal Value"]
        else:
            field_names = ["Parameters"] + ["Optimal Value %d" % i for i in range(1, len(candidates) + 1)]
        table = PrettyTable(field_names=field_names, float_format=".6", align="l")
        # add parameters
        for param in parameters:
            row = [param] + [obs.config.get_dictionary().get(param) for obs in candidates]
            table.add_row(row)
        # add objectives
        if self.num_objectives == 1:
            table.add_row(["Optimal Objective Value"] + [obs.objectives[0] for obs in candidates])
        else:
            for i in range(self.num_objectives):
                table.add_row([f"Objective {i+1}"] + [obs.objectives[i] for obs in candidates])
        # add constraints
        if self.num_constraints > 0:
            for i in range(self.num_constraints):
                table.add_row([f"Constraint {i+1}"] + [obs.constraints[i] for obs in candidates])
        # add other info
        row = ["Num Trials", len(self)]
        if n_candidates >= 3 and max_candidates >= 3:
            row += ["Num Best" if self.num_objectives == 1 else "Num Pareto", n_candidates] + [""] * (len(candidates)-3)
        else:
            row += [""] * (len(candidates) - 1)
        table.add_row(row)

        # add hlines for the last result rows
        n_last_rows = 1  # rows for other info
        raw_table = str(table)
        lines = raw_table.splitlines()
        hline = lines[2]
        # add a hline before objectives
        lines.insert(3 + len(parameters), hline)
        # add a hline before constraints
        if self.num_constraints > 0:
            lines.insert(4 + len(parameters) + self.num_objectives, hline)
        # add hlines for other info
        for i in range(n_last_rows):
            lines.insert(-(i + 1) * 2, hline)
        render_table = "\n".join(lines)
        return render_table  # type: str

    def __str__(self):
        return self.get_str()

    __repr__ = __str__

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

        config_space = self.get_config_space()
        parameters = list(config_space.get_hyperparameter_names())

        if method == 'fanova':
            importance_func = partial(get_fanova_importance, config_space=config_space)
        elif method == 'shap':
            importance_func = get_shap_importance
            if any([isinstance(hp, (CategoricalHyperparameter, OrdinalHyperparameter))
                    for hp in config_space.get_hyperparameters()]):
                logger.warning("SHAP can not support categorical/ordinal hyperparameters well. "
                               "To analyze a space with categorical/ordinal hyperparameters, "
                               "we recommend setting the method to fanova.")
        else:
            raise ValueError("Invalid method for feature importance: %s" % method)

        X = self.get_config_array(transform='numerical')
        Y = self.get_objectives(transform='failed')
        cY = self.get_constraints(transform='failed,bilog')

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
        return importance_table  # type: PrettyTable  # the table can be printed directly

    def plot_convergence(self, true_minimum=None, name=None, clip_y=True,
                         title="Convergence plot", xlabel="Iteration", ylabel="Min objective value",
                         ax=None, alpha=0.3, yscale=None, color='C0', infeasible_color='C1', **kwargs):
        """
        Plot convergence trace.

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
        if self.num_objectives > 1:
            raise ValueError('plot_convergence only supports single-objective optimization. '
                             'Please use plot_pareto_front or plot_hypervolumes instead.')

        y = self.get_objectives(transform='failed').reshape(-1)  # do not transform infeasible trials
        cy = self.get_constraints(transform='none', warn_invalid_value=False)
        ax = plot_convergence(y, cy, true_minimum, name, clip_y, title, xlabel, ylabel, ax, alpha, yscale,
                              color, infeasible_color, **kwargs)
        return ax

    def plot_pareto_front(self, title="Pareto Front", ax=None, alpha=0.3, color='C0', infeasible_color='C1', **kwargs):
        """
        Plot Pareto front

        Parameters
        ----------
        see `plot_pareto_front` in `openbox.visualization`.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The matplotlib axes.
        """
        from openbox.visualization import plot_pareto_front
        assert self.num_objectives > 1
        if self.num_objectives not in [2, 3]:
            raise ValueError('plot_pareto_front only supports 2 or 3 objectives!')

        y = self.get_objectives(transform='failed')  # do not transform infeasible trials
        cy = self.get_constraints(transform='none', warn_invalid_value=False)
        ax = plot_pareto_front(y, cy, title, ax, alpha, color, infeasible_color, **kwargs)
        return ax

    def plot_hypervolumes(self, optimal_hypervolume=None, ref_point=None, logy=False, ax=None, **kwargs):
        """
        Plot the hypervolume of the Pareto front over time.

        Parameters
        ----------
        optimal_hypervolume : float, optional
            The optimal hypervolume. If provided, plot the hypervolume difference.

        ref_point : List[float], optional
            Reference point for hypervolume calculation. If None, use self.ref_point.

        logy : bool, default=False
            Whether to plot the hypervolume on log base 10 scale.

        For other parameters, see `plot_curve` in `openbox.visualization`.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The matplotlib axes.
        """
        from openbox.visualization import plot_curve

        assert self.num_objectives > 1
        ref_point = self.check_ref_point(ref_point)
        ref_point = ref_point if ref_point is not None else self.ref_point
        assert ref_point is not None, 'ref_point must be provided!'

        x = np.arange(len(self)) + 1
        y = self.compute_hypervolume(ref_point=ref_point, data_range='all')
        y = np.asarray(y, dtype=np.float64)
        if optimal_hypervolume is not None:
            y = optimal_hypervolume - y
            ylabel = 'Hypervolume Difference'
        else:
            ylabel = 'Hypervolume'
        if logy:
            ylabel = 'Log ' + ylabel
            y = np.log10(y)  # log base 10  # todo: handle 0
        xlabel = 'Iteration'
        ax = plot_curve(x=x, y=y, xlabel=xlabel, ylabel=ylabel, ax=ax, **kwargs)
        return ax

    def visualize_html(self, logging_dir='logs/', open_html=True, show_importance=False, verify_surrogate=False,
                       task_info=None, optimizer=None, advisor=None, **kwargs):
        """
        Visualize the history using OpenBox's HTML visualization.

        Parameters
        ----------
        logging_dir : str, default='logs/'
            The directory to save the visualization.
        open_html: bool, default=True
            If True, the visualization will be opened in the browser automatically.
        show_importance: bool, default=False
            If True, the importance of each hyperparameter will be calculated and shown.
            Note that additional packages are required to calculate the importance. (run `pip install shap lightgbm`)
        verify_surrogate: bool, default=False
            If True, the surrogate model will be verified and shown. This may take some time.
        task_info : dict, optional
            Task information for visualizer to use.
        optimizer : Optimizer, optional
            Optimizer to extract task_info from.
        advisor : Advisor, optional
            Advisor to extract task_info from.
        kwargs: dict
            Other keyword arguments passed to `build_visualizer` in `openbox.visualization`.
        """
        from openbox.visualization import build_visualizer, HTMLVisualizer

        option = 'advanced' if (show_importance or verify_surrogate) else 'basic'
        visualizer = build_visualizer(
            option=option, history=self, logging_dir=logging_dir,
            task_info=task_info, optimizer=optimizer, advisor=advisor, **kwargs)  # type: HTMLVisualizer
        if visualizer.history is not self:
            visualizer.history = self
            visualizer.meta_data['task_id'] = self.task_id
        visualizer.visualize(open_html=open_html, show_importance=show_importance, verify_surrogate=verify_surrogate)
        return visualizer

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
        y = self.get_objectives(transform='none', warn_invalid_value=True)  # todo: check if has invalid value
        cy = self.get_constraints(transform='none', warn_invalid_value=True)
        exp = visualize_hiplot(configs=configs, y=y, cy=cy, html_file=html_file, **kwargs)
        return exp

    def save_json(self, filename: str):
        dirname = os.path.dirname(filename)
        if dirname != '' and not os.path.exists(dirname):
            logger.info(f'Creating directory to save history: {dirname}')
            os.makedirs(dirname, exist_ok=True)

        # todo: save rng, config space with random state into json
        data = {
            'task_id': self.task_id,
            'num_objectives': self.num_objectives,
            'num_constraints': self.num_constraints,
            # 'config_space': self.config_space,
            'ref_point': self.ref_point,
            'meta_info': self.meta_info,
            'global_start_time': self.global_start_time.isoformat(),
            'observations': [
                obs.to_dict() for obs in self.observations
            ]
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f'Saved history (len={len(self)}) to {filename}')

    @classmethod
    def load_json(cls, filename: str, config_space: ConfigurationSpace) -> 'History':
        if not os.path.exists(filename):
            raise FileNotFoundError(f'File not found: {filename}')
        with open(filename, 'r') as f:
            data = json.load(f)

        # todo: load rng, config space with random state from json
        global_start_time = data.pop('global_start_time')
        global_start_time = datetime.fromisoformat(global_start_time)
        observations = data.pop('observations')
        observations = [Observation.from_dict(obs, config_space) for obs in observations]

        history = cls(**data)
        history.global_start_time = global_start_time
        history.update_observations(observations)

        logger.info(f'Loaded history (len={len(observations)}) from {filename}')
        return history


class MultiStartHistory(History):
    """
    History for multi-start algorithms.

    TODO: All methods return the result of current start. Consider returning the result of all starts.
    """
    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stored_observations = []  # type: List[List[Observation]]

    def restart(self):
        self.stored_observations.append(self.observations)
        self.observations = []

    def get_observations_for_all_restarts(self):
        return [obs for obs_list in self.stored_observations for obs in obs_list] + self.observations
