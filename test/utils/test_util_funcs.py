import pytest
import numpy as np
from openbox.utils.config_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant, \
    OrdinalHyperparameter
from openbox.utils.util_funcs import get_types, transform_to_1d_list, parse_result, check_random_state, get_rng, deprecate_kwarg


def test_get_types_with_multiple_hyperparameters():
    config_space = ConfigurationSpace()
    config_space.add_hyperparameter(OrdinalHyperparameter("param1", [1, 2, 3]))
    config_space.add_hyperparameter(CategoricalHyperparameter("param2", ["option1", "option2"]))
    config_space.add_hyperparameter(UniformIntegerHyperparameter("param3", 1, 10))
    config_space.add_hyperparameter(UniformFloatHyperparameter("param4", 0.0, 1.0))
    config_space.add_hyperparameter(Constant("param5", 2))
    types, bounds = get_types(config_space)
    assert len(types) == 5
    assert len(bounds) == 5


def test_transform_to_1d_list_with_2d_input():
    X = np.array([[1, 2], [3, 4]])
    with pytest.raises(AssertionError):
        transform_to_1d_list(X)

    with pytest.raises(AssertionError):
        transform_to_1d_list(None)

    X = np.array([[1], [3]])
    assert transform_to_1d_list(X, hint='result') == [1, 3]


def test_parse_result_with_constraints_and_extra_info():
    result = {'objectives': [1.0], 'constraints': [0.0], 'extra_info': {'key': 'value'}}
    objectives, constraints, extra_info = parse_result(result)
    assert objectives == [1.0]
    assert constraints == [0.0]
    assert extra_info == {'key': 'value'}


def test_check_random_state_with_random_state_input():
    rng = check_random_state(None)
    assert isinstance(rng, np.random.RandomState)

    rng = check_random_state(np.random)
    assert isinstance(rng, np.random.RandomState)

    rng = check_random_state(0)
    assert isinstance(rng, np.random.RandomState)

    rng = np.random.RandomState(0)
    rng_returned = check_random_state(rng)
    assert rng_returned is rng

    with pytest.raises(ValueError, match='.*cannot be used to seed a numpy.random.RandomState.*'):
        check_random_state('invalid_seed')


def test_get_rng_with_rng_and_run_id():
    run_id, rng = get_rng(None, None)
    assert isinstance(run_id, int)
    assert isinstance(rng, np.random.RandomState)

    run_id, rng = get_rng(None, 1)
    assert run_id == 1
    assert isinstance(rng, np.random.RandomState)

    run_id, rng = get_rng(1, None)
    assert run_id == 1
    assert isinstance(rng, np.random.RandomState)

    rng_input = np.random.RandomState(1)
    run_id, rng = get_rng(rng_input, None)
    assert isinstance(run_id, int)
    assert rng is rng_input

    rng = np.random.RandomState(0)
    run_id = 1
    run_id_returned, rng_returned = get_rng(rng, run_id)
    assert run_id_returned == run_id
    assert rng_returned is rng

    with pytest.raises(TypeError):
        get_rng('invalid', None)

    with pytest.raises(TypeError):
        get_rng(None, 'invalid')


@deprecate_kwarg('old_kwarg', 'new_kwarg')
def function_with_deprecated_kwarg_decorated(old_kwarg=None, new_kwarg=None):
    return old_kwarg, new_kwarg


def test_deprecate_kwarg():
    with pytest.raises(TypeError):
        function_with_deprecated_kwarg_decorated(old_kwarg='old', new_kwarg='new')

    old_value, new_value = function_with_deprecated_kwarg_decorated(old_kwarg='old')
    assert old_value is None
    assert new_value == 'old'

    old_value, new_value = function_with_deprecated_kwarg_decorated(new_kwarg='new')
    assert old_value is None
    assert new_value == 'new'

    old_value, new_value = function_with_deprecated_kwarg_decorated()
    assert old_value is None
    assert new_value is None
