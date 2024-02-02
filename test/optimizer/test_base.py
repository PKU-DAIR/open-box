import pytest
from unittest.mock import MagicMock, patch
from openbox.optimizer.base import BOBase
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.history import History, Observation


def test_bo_base_initialization(configspace_tiny, func_brain):
    config_space = configspace_tiny
    objective_function = func_brain
    bo_base = BOBase(objective_function, config_space)
    assert bo_base.config_space == config_space
    assert bo_base.objective_function == objective_function
    assert bo_base.init_num == 3
    assert bo_base.max_runs == 100
    assert bo_base.max_runtime == float('inf')
    assert bo_base.time_left == float('inf')
    assert bo_base.max_runtime_per_trial is None
    assert bo_base.iteration_id == 0
    assert bo_base.sample_strategy == 'bo'
    assert bo_base.transfer_learning_history is None
    assert bo_base.config_advisor is None

    with pytest.raises(NotImplementedError):
        bo_base.run()

    with pytest.raises(NotImplementedError):
        bo_base.iterate()

    with pytest.raises(AssertionError):
        bo_base.get_history()

    with pytest.raises(AssertionError):
        bo_base.get_incumbents()

