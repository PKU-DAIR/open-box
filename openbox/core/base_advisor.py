import os
import abc
import numpy as np
from datetime import datetime
from typing import List
from ConfigSpace import Configuration, ConfigurationSpace

from openbox import logger
from openbox.utils.util_funcs import check_random_state, deprecate_kwarg
from openbox.utils.early_stop import EarlyStopAlgorithm, EarlyStopException
from openbox.utils.history import Observation, History
from openbox.utils.constants import MAXINT
from openbox.core.space_adapter import IdentitySpaceAdapter, CompressorSpaceAdapter


class BaseAdvisor(object, metaclass=abc.ABCMeta):
    """
    Base Advisor Class.
      Implement get_suggestion() to get a new configuration suggestion.
      Call update_observation() to update the advisor with a new observation.

    Parameters
    ----------
    config_space: ConfigurationSpace
        Configuration space object.
    num_objectives: int, default=1
        Number of objectives.
    num_constraints: int, default=0
        Number of constraints.
    ref_point: optional, list or np.ndarray
        Reference point for hypervolume calculation in multi-objective optimization.
    output_dir: str
        Output directory.
    task_id: str
        Task id.
    random_state: optional, int or np.random.RandomState
        Random state.
    logger_kwargs: optional, dict
        Additional arguments for logger.
    """

    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    def __init__(
            self,
            config_space,
            num_objectives=1,
            num_constraints=0,
            ref_point=None,
            early_stop=False,
            early_stop_kwargs=None,
            output_dir='logs',
            task_id='OpenBox',
            random_state=None,
            logger_kwargs: dict = None,
            compressor=None,
            compressor_type='none',
            compressor_kwargs=None,
    ):

        self.timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        self.num_objectives = num_objectives
        self.num_constraints = num_constraints
        self.output_dir = output_dir
        self.task_id = task_id
        self.rng = check_random_state(random_state)

        _logger_kwargs = {'name': task_id, 'logdir': output_dir}
        _logger_kwargs.update(logger_kwargs or {})
        logger.init(**_logger_kwargs)

        self.config_space = config_space
        self.config_space_seed = self.rng.randint(MAXINT)
        self.config_space.seed(self.config_space_seed)
        self.ref_point = ref_point
        self.sample_space = self.config_space
        self.surrogate_space = self.config_space

        # space compression setting
        if compressor is not None:
            self.space_adapter = CompressorSpaceAdapter(
                config_space=self.config_space,
                compressor=compressor,
            )
        elif compressor_type is not None and str(compressor_type).lower() == 'none':
            logger.info('compressor_type=none, using identity space adapter.')
            self.space_adapter = IdentitySpaceAdapter(self.config_space)
        elif compressor_type is not None or compressor_kwargs is not None:
            compressor = self._build_compressor(
                config_space=self.config_space,
                compressor_type=compressor_type,
                compressor_kwargs=compressor_kwargs,
                seed=self.config_space_seed,
            )
            self.space_adapter = CompressorSpaceAdapter(
                config_space=self.config_space,
                compressor=compressor,
            )
        else:
            logger.info('No compressor is provided, using identity space adapter.')
            self.space_adapter = IdentitySpaceAdapter(self.config_space)

        # init history
        self.history = History(
            task_id=task_id, num_objectives=num_objectives, num_constraints=num_constraints, config_space=config_space,
            ref_point=ref_point, meta_info=None,  # todo: add meta info
        )


        # early stop
        self.early_stop = early_stop
        early_stop_kwargs = early_stop_kwargs or dict()
        self.early_stop_algorithm = EarlyStopAlgorithm(**early_stop_kwargs) if self.early_stop else None
        if self.early_stop:
            logger.info(f'Early stop is enabled.')

    @staticmethod
    def _build_compressor(config_space, compressor_type, compressor_kwargs, seed):
        from openbox.compressor import Compressor
        from openbox.compressor.api import (
            create_steps_from_strings,
            create_filling_from_config,
            create_filling_from_string,
        )

        kwargs = dict(compressor_kwargs or {})
        kwargs.setdefault('seed', seed)

        step_params = kwargs.pop('step_params', {})
        filling_config = kwargs.pop('filling_config', None)
        filling_type = kwargs.pop('filling_type', None)
        fixed_values = kwargs.pop('fixed_values', None)

        filling_strategy = None
        if filling_config is not None:
            filling_strategy = create_filling_from_config(filling_config)
        elif filling_type is not None or fixed_values is not None:
            filling_strategy = create_filling_from_string(
                filling_str=filling_type or 'default',
                fixed_values=fixed_values,
            )

        raw_steps = kwargs.pop('steps', None)
        if raw_steps is not None:
            if len(raw_steps) > 0 and isinstance(raw_steps[0], str):
                steps = create_steps_from_strings(raw_steps, step_params=step_params)
            else:
                steps = raw_steps
            return Compressor(
                config_space=config_space,
                steps=steps,
                filling_strategy=filling_strategy,
                **kwargs,
            )

        step_strings = kwargs.pop('step_strings', None)
        if step_strings is not None:
            steps = create_steps_from_strings(step_strings, step_params=step_params)
            return Compressor(
                config_space=config_space,
                steps=steps,
                filling_strategy=filling_strategy,
                **kwargs,
            )

        compressor_type = (compressor_type or 'none').lower()
        step_strings = []
        mapped_step_params = {}

        if compressor_type == 'none':
            step_strings = []
        elif compressor_type == 'pipeline':
            raise ValueError('compressor_type="pipeline" requires `steps` or `step_strings` in compressor_kwargs.')
        elif compressor_type in ('shap', 'expert'):
            if compressor_type == 'shap':
                step_strings.append('d_shap')
                mapped_step_params['d_shap'] = {
                    'topk': kwargs.pop('topk', 20),
                    'exclude_params': kwargs.pop('exclude_params', None),
                }
            else:
                step_strings.append('d_expert')
                mapped_step_params['d_expert'] = {
                    'expert_params': kwargs.pop('expert_params', []),
                    'exclude_params': kwargs.pop('exclude_params', None),
                }
            top_ratio = kwargs.pop('top_ratio', 0.8)
            sigma = kwargs.pop('sigma', 2.0)
            if top_ratio < 1.0 or sigma > 0:
                step_strings.append('r_boundary')
                mapped_step_params['r_boundary'] = {
                    'top_ratio': top_ratio,
                    'sigma': sigma,
                    'enable_mixed_sampling': kwargs.pop('enable_mixed_sampling', True),
                    'initial_prob': kwargs.pop('initial_prob', 0.9),
                }
        elif compressor_type == 'llamatune':
            max_num_values = kwargs.pop('max_num_values', None)
            adapter_alias = kwargs.pop('adapter_alias', 'none')
            low_dim = kwargs.pop('le_low_dim', 10)

            if max_num_values is not None:
                step_strings.append('p_quant')
                mapped_step_params['p_quant'] = {'max_num_values': max_num_values, 'seed': kwargs.get('seed', seed)}
            if adapter_alias == 'rembo':
                step_strings.append('p_rembo')
                mapped_step_params['p_rembo'] = {
                    'low_dim': low_dim,
                    'max_num_values': max_num_values,
                    'seed': kwargs.get('seed', seed),
                }
            elif adapter_alias == 'hesbo':
                step_strings.append('p_hesbo')
                mapped_step_params['p_hesbo'] = {
                    'low_dim': low_dim,
                    'max_num_values': max_num_values,
                    'seed': kwargs.get('seed', seed),
                }
            elif adapter_alias != 'none':
                raise ValueError(f'Unknown adapter_alias: {adapter_alias}.')
        else:
            raise ValueError(f'Unknown compressor_type: {compressor_type}.')

        step_params = {**mapped_step_params, **step_params}
        steps = create_steps_from_strings(step_strings, step_params=step_params)
        return Compressor(
            config_space=config_space,
            steps=steps,
            filling_strategy=filling_strategy,
            **kwargs,
        )

    def setup_space_adapter(self, transfer_learning_history=None):
        transformed_history = self.space_adapter.setup(transfer_learning_history)
        self.sample_space = self.space_adapter.sample_space
        self.surrogate_space = self.space_adapter.surrogate_space
        self.sample_space.seed(self.config_space_seed)
        self.surrogate_space.seed(self.config_space_seed)
        return transformed_history

    def early_stop_perf(self, history):
        if not self.early_stop:
            return

        if self.early_stop_algorithm.already_early_stopped(history):
            raise EarlyStopException("Early stop triggered!")

        if self.early_stop_algorithm.decide_early_stop_before_suggest(history):
            self.early_stop_algorithm.set_already_early_stopped(history)
            raise EarlyStopException("Early stop triggered!")

    def get_suggestion(self, *args, **kwargs) -> Configuration:
        """
        Get a suggestion for the next configuration to evaluate.

        Parameters
        ----------
        args, kwargs
            Additional arguments and named arguments.

        Returns
        -------
        config: Configuration
            The next configuration to evaluate.
        """
        raise NotImplementedError

    def get_suggestions(self, *args, **kwargs) -> List[Configuration]:
        """
        Get a list of suggestions for the next configurations to evaluate.

        Parameters
        ----------
        args, kwargs
            Additional arguments and named arguments.

        Returns
        -------
        configs: List[Configuration]
            A list of configurations to evaluate.
        """
        raise NotImplementedError

    def update_observation(self, observation: Observation):
        """
        Update the advisor with a new observation.

        Parameters
        ----------
        observation: Observation
            Observation of the objective function.
        """
        self.space_adapter.cache_observation(observation)   # low_dim_config
        return self.history.update_observation(observation)

    def update_observations(self, observations: List[Observation]):
        """
        Update the advisor with a new batch of observations.

        Parameters
        ----------
        observations: List[Observation]
            Observations of the objective function.
        """
        for observation in observations:
            self.update_observation(observation)

    @staticmethod
    def sample_random_configs(config_space, num_configs=1, excluded_configs=None):
        """
        Sample a batch of random configurations.

        Parameters
        ----------
        config_space: ConfigurationSpace
            Configuration space object.
        num_configs: int
            Number of configurations to sample.
        excluded_configs: optional, List[Configuration] or Set[Configuration]
            A list of excluded configurations.

        Returns
        -------
        configs: List[Configuration]
            A list of sampled configurations.
        """
        if excluded_configs is None:
            excluded_configs = set()

        configs = list()
        sample_cnt = 0
        max_sample_cnt = 1000
        while len(configs) < num_configs:
            config = config_space.sample_configuration()
            sample_cnt += 1
            if config not in configs and config not in excluded_configs:
                configs.append(config)
                sample_cnt = 0
                continue
            if sample_cnt >= max_sample_cnt:
                logger.warning('Cannot sample non duplicate configuration after %d iterations.' % max_sample_cnt)
                configs.append(config)
                sample_cnt = 0
        return configs

    def get_history(self):
        """
        Get the history of the advisor.

        Returns
        -------
        history: History
            History of the advisor.
        """
        return self.history

    def save_json(self, filename: str = None):
        """
        Save history to a json file.

        Parameters
        ----------
        filename: str
            Filename to save history.
        """
        if filename is None:
            filename = os.path.join(self.output_dir, f'history/{self.task_id}/history_{self.timestamp}.json')
        self.history.save_json(filename)

    def load_json(self, filename: str):
        """
        Load history from a json file.

        Parameters
        ----------
        filename: str
            Filename to load history.
        """
        self.history = History.load_json(filename, self.config_space)
