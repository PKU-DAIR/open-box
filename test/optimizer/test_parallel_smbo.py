import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from openbox.optimizer.parallel_smbo import pSMBO, wrapper
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.history import History, Observation
from openbox.utils.constants import MAXINT, SUCCESS

objective_function = None


def branin(config):
    x1, x2 = config['x1'], config['x2']
    y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2 \
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    return {'objectives': [y]}


def test_psmbo_sync(configspace_tiny):
    config_space = configspace_tiny
    global objective_function
    objective_function = branin
    psmbo = pSMBO(objective_function, config_space, parallel_strategy='sync', initial_runs=1, max_runs=2)
    assert psmbo.config_space == config_space
    assert psmbo.objective_function == objective_function
    assert psmbo.parallel_strategy == 'sync'
    assert psmbo.batch_size == 4
    assert psmbo.sample_strategy == 'bo'
    assert psmbo.init_num == 1
    assert psmbo.output_dir == 'logs'
    assert psmbo.task_id == 'OpenBox'
    assert psmbo.rng is not None
    assert psmbo.config_advisor is not None

    psmbo.sync_run()
    history = psmbo.run()
    assert len(history) > 0


def test_psmbo_async(configspace_tiny):
    config_space = configspace_tiny
    global objective_function
    objective_function = branin
    psmbo = pSMBO(objective_function, config_space, parallel_strategy='async', initial_runs=1, max_runs=2)
    assert psmbo.config_space == config_space
    assert psmbo.objective_function == objective_function
    assert psmbo.parallel_strategy == 'async'
    assert psmbo.batch_size == 4
    assert psmbo.sample_strategy == 'bo'
    assert psmbo.init_num == 1
    assert psmbo.output_dir == 'logs'
    assert psmbo.task_id == 'OpenBox'
    assert psmbo.rng is not None
    assert psmbo.config_advisor is not None

    psmbo.async_run()
    history = psmbo.run()
    assert len(history) > 0

    observations = psmbo.async_iterate(n=2)
    assert len(observations) == 2



#
#
# def test_psmbo_run():
#     config_space = mock_config_space()
#     objective_function = mock_objective_function()
#     psmbo = pSMBO(objective_function, config_space)
#     with patch.object(pSMBO, 'async_run') as mock_async_run, patch.object(pSMBO, 'sync_run') as mock_sync_run:
#         psmbo.run()
#         mock_async_run.assert_called_once()
#         mock_sync_run.assert_not_called()
#         psmbo.parallel_strategy = 'sync'
#         psmbo.run()
#         mock_sync_run.assert_called_once()
#
#
# def test_wrapper():
#     objective_function = mock_objective_function()
#     config = MagicMock()
#     timeout = 1
#     FAILED_PERF = float('inf')
#     observation = wrapper((objective_function, config, timeout, FAILED_PERF))
#     assert isinstance(observation, Observation)
