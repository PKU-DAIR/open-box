import os
import abc
import numpy as np
from datetime import datetime
from typing import List
from ConfigSpace import Configuration, ConfigurationSpace

from openbox import logger
from openbox.utils.util_funcs import check_random_state, deprecate_kwarg
from openbox.utils.history import Observation, History
from openbox.utils.constants import MAXINT


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
            output_dir='logs',
            task_id='OpenBox',
            random_state=None,
            logger_kwargs: dict = None,
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

        # init history
        self.history = History(
            task_id=task_id, num_objectives=num_objectives, num_constraints=num_constraints, config_space=config_space,
            ref_point=ref_point, meta_info=None,  # todo: add meta info
        )

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
