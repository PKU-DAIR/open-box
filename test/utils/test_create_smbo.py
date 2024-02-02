import pytest
from openbox.optimizer import _optimizers
from openbox.utils.config_space.space_utils import get_config_space_from_dict
from openbox.utils.start_smbo import create_smbo


def test_create_smbo_with_valid_input():
    def objective_func(x):
        return x ** 2

    kwargs = {
        'optimizer': 'SMBO',
        'parameters': {'x': {'type': 'int', 'bound': [1, 10], 'default': 5, 'log': 'false', 'q': 1}},
        'conditions': {}
    }
    smbo = create_smbo(objective_func, **kwargs)
    assert smbo is not None


def test_create_smbo_with_invalid_optimizer():
    def objective_func(x):
        return x ** 2

    kwargs = {
        'optimizer': 'InvalidOptimizer',
        'parameters': {'x': {'type': 'int', 'bound': [1, 10], 'default': 5, 'log': 'false', 'q': 1}},
        'conditions': {}
    }
    with pytest.raises(KeyError):
        create_smbo(objective_func, **kwargs)


def test_create_smbo_without_optimizer():
    def objective_func(x):
        return x ** 2

    kwargs = {
        'parameters': {'x': {'type': 'int', 'bound': [1, 10], 'default': 5, 'log': 'false', 'q': 1}},
        'conditions': {}
    }
    with pytest.raises(KeyError):
        create_smbo(objective_func, **kwargs)


def test_create_smbo_without_parameters():
    def objective_func(x):
        return x ** 2

    kwargs = {
        'optimizer': 'SMBO',
        'conditions': {}
    }
    with pytest.raises(KeyError):
        create_smbo(objective_func, **kwargs)
