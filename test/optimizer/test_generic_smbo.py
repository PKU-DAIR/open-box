import pytest
from unittest.mock import MagicMock, patch
from openbox.optimizer.base import BOBase
from openbox.optimizer.generic_smbo import SMBO
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.history import History, Observation


def test_smbo(configspace_tiny, func_brain):
    config_space = configspace_tiny
    objective_function = func_brain
    smbo = SMBO(objective_function, config_space, max_runs=2, initial_runs=1, logging_dir='test/datas')
    assert isinstance(smbo, BOBase)
    assert smbo.objective_function == objective_function
    assert smbo.config_space == config_space

    smbo.run()
    assert smbo.iteration_id == 2
    assert len(smbo.config_advisor.history) == 2
