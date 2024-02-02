import pytest
from unittest.mock import MagicMock
from openbox.acquisition_function.acquisition import AbstractAcquisitionFunction
from openbox.utils.config_space import ConfigurationSpace, Configuration
from openbox.utils.history import History
from openbox.utils.constants import MAXINT
from openbox.acquisition_function.acquisition import EI
from openbox.acq_optimizer import AcquisitionFunctionMaximizer, CMAESMaximizer, LocalSearchMaximizer, \
    RandomSearchMaximizer, InterleavedLocalAndRandomSearchMaximizer, ScipyMaximizer, RandomScipyMaximizer, \
    ScipyGlobalMaximizer, StagedBatchScipyMaximizer, MESMO_Maximizer, USeMO_Maximizer, batchMCMaximizer
import numpy as np


def test_acquisition_function_maximizer_initialization(configspace_tiny):
    config_space = configspace_tiny
    with pytest.raises(TypeError, match="Can't instantiate abstract class AcquisitionFunctionMaximizer.*"):
        AcquisitionFunctionMaximizer(config_space)


def test_cmaes_maximizer_initialization(configspace_tiny, acq_func_ei, history_single_obs):
    config_space = configspace_tiny
    acquisition_function = acq_func_ei
    runhistory = history_single_obs

    maximizer = CMAESMaximizer(config_space)
    assert maximizer.config_space == config_space
    assert isinstance(maximizer.rng, np.random.RandomState)

    challengers = maximizer.maximize(acquisition_function, runhistory, 4)
    assert len(challengers) >= 4


def test_local_search_maximizer_initialization(configspace_tiny, acq_func_ei, history_single_obs):
    config_space = configspace_tiny
    acquisition_function = acq_func_ei
    runhistory = history_single_obs

    maximizer = LocalSearchMaximizer(config_space)
    assert maximizer.config_space == config_space
    assert isinstance(maximizer.rng, np.random.RandomState)

    challengers = maximizer.maximize(acquisition_function, runhistory, 3)
    assert len(challengers) == 3

    # test when runhistory is empty
    runhistory.observations = []
    challengers = maximizer.maximize(acquisition_function, runhistory, 3)
    assert len(challengers) == 3


def test_random_search_maximizer_initialization(configspace_tiny, acq_func_ei, history_single_obs):
    config_space = configspace_tiny
    acquisition_function = acq_func_ei
    runhistory = history_single_obs

    maximizer = RandomSearchMaximizer(config_space)
    assert maximizer.config_space == config_space
    assert isinstance(maximizer.rng, np.random.RandomState)

    challengers = maximizer.maximize(acquisition_function, runhistory, 3)
    assert len(challengers) == 3

    # test when runhistory is empty
    runhistory.observations = []
    challengers = maximizer.maximize(acquisition_function, runhistory, 3, _sort=True)
    assert len(challengers) == 3


def test_interleaved_local_and_random_search_maximizer(configspace_tiny, acq_func_ei, history_single_obs):
    config_space = configspace_tiny
    acquisition_function = acq_func_ei
    runhistory = history_single_obs

    maximizer = InterleavedLocalAndRandomSearchMaximizer(config_space, n_sls_iterations=2)
    assert maximizer.config_space == config_space
    assert isinstance(maximizer.rng, np.random.RandomState)

    challengers = maximizer.maximize(acquisition_function, runhistory, 3)
    assert len(challengers) == 3

    # test when runhistory is empty
    runhistory.observations = []
    challengers = maximizer.maximize(acquisition_function, runhistory, 3, _sort=True)
    assert len(challengers) == 3


def test_scipy_search_maximizer(configspace_tiny, acq_func_ei, history_single_obs):
    config_space = configspace_tiny
    acquisition_function = acq_func_ei
    runhistory = history_single_obs

    maximizer = ScipyMaximizer(config_space)
    assert maximizer.config_space == config_space
    assert isinstance(maximizer.rng, np.random.RandomState)

    challengers = maximizer.maximize(acquisition_function, runhistory)
    assert len(challengers) == 1


def test_random_scipy_search_maximizer(configspace_tiny, acq_func_ei, history_single_obs):
    config_space = configspace_tiny
    acquisition_function = acq_func_ei
    runhistory = history_single_obs

    maximizer = RandomScipyMaximizer(config_space)
    assert maximizer.config_space == config_space
    assert isinstance(maximizer.rng, np.random.RandomState)

    challengers = maximizer.maximize(acquisition_function, runhistory, 3)
    assert len(challengers) >= 3


def test_scipy_global_maximizer(configspace_tiny, acq_func_ei, history_single_obs):
    config_space = configspace_tiny
    acquisition_function = acq_func_ei
    runhistory = history_single_obs

    maximizer = ScipyGlobalMaximizer(config_space)
    assert maximizer.config_space == config_space
    assert isinstance(maximizer.rng, np.random.RandomState)

    challengers = maximizer.maximize(acquisition_function, runhistory)
    assert len(challengers) == 1


def test_staged_batch_scipy_maximizer(configspace_tiny, acq_func_ei, history_single_obs):
    config_space = configspace_tiny
    acquisition_function = acq_func_ei
    runhistory = history_single_obs

    maximizer = StagedBatchScipyMaximizer(config_space)
    assert maximizer.config_space == config_space
    assert isinstance(maximizer.rng, np.random.RandomState)

    challengers = maximizer.maximize(acquisition_function, runhistory, num_points=3)
    assert len(challengers) >= 3


def test_mesmo_maximizer(configspace_tiny, acq_func_ei, history_single_obs):
    config_space = configspace_tiny
    acquisition_function = acq_func_ei
    runhistory = history_single_obs

    maximizer = MESMO_Maximizer(config_space)
    assert maximizer.config_space == config_space
    assert isinstance(maximizer.rng, np.random.RandomState)

    challengers = maximizer.maximize(acquisition_function, runhistory, num_points=3)
    assert len(challengers) >= 3


def test_usemo_maximizer(configspace_tiny, acq_func_usemo, history_double_obs):
    config_space = configspace_tiny
    acquisition_function = acq_func_usemo
    runhistory = history_double_obs

    maximizer = USeMO_Maximizer(config_space)
    assert maximizer.config_space == config_space
    assert isinstance(maximizer.rng, np.random.RandomState)

    challengers = maximizer.maximize(acquisition_function, runhistory, num_points=3)
    assert len(challengers) >= 3


def test_batchMC_maximizer(configspace_tiny, acq_func_ei, history_single_obs):
    config_space = configspace_tiny
    acquisition_function = acq_func_ei
    runhistory = history_single_obs

    maximizer = batchMCMaximizer(config_space)
    assert maximizer.config_space == config_space
    assert isinstance(maximizer.rng, np.random.RandomState)

    challengers = maximizer.maximize(acquisition_function, runhistory, num_points=2)
    assert len(challengers) >= 2