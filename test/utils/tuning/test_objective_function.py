import pytest
from openbox.utils.tuning.objective_function import get_objective_function, get_lightgbm_objective_function, \
    get_xgboost_objective_function


@pytest.fixture
def get_x_y_train_val():
    return None, None, None, None


def test_get_objective_function_lightgbm_cls(get_x_y_train_val):
    x_train, x_val, y_train, y_val = get_x_y_train_val
    # Test for model 'lightgbm' and task_type 'cls'
    objective_function = get_objective_function('lightgbm', x_train, x_val, y_train, y_val, 'cls')
    assert callable(objective_function)  # You may want to add more specific checks for the returned function

    # Test for model 'xgboost' and task_type 'cls'
    objective_function = get_objective_function('xgboost', x_train, x_val, y_train, y_val, 'cls')
    assert callable(objective_function)  # You may want to add more specific checks for the returned function

    # Test for an unsupported model
    with pytest.raises(ValueError, match='Unsupported model: invalid_model.'):
        get_objective_function('invalid_model', x_train, x_val, y_train, y_val, 'cls')


def test_get_lightgbm_objective_function(get_x_y_train_val):
    x_train, x_val, y_train, y_val = get_x_y_train_val
    # Test for task_type 'cls'
    objective_function = get_lightgbm_objective_function(x_train, x_val, y_train, y_val, 'cls')
    assert callable(objective_function)

    # Test for task_type 'rgs'
    with pytest.raises(NotImplementedError):
        get_lightgbm_objective_function(x_train, x_val, y_train, y_val, 'rgs')

    # Test for an unsupported task_type
    with pytest.raises(ValueError, match='Unsupported task type: invalid_type.'):
        get_lightgbm_objective_function(x_train, x_val, y_train, y_val, 'invalid_type')


def test_get_xgboost_objective_function(get_x_y_train_val):
    x_train, x_val, y_train, y_val = get_x_y_train_val
    # Test for task_type 'cls'
    objective_function = get_xgboost_objective_function(x_train, x_val, y_train, y_val, 'cls')
    assert callable(objective_function)

    # Test for task_type 'rgs'
    with pytest.raises(NotImplementedError):
        get_xgboost_objective_function(x_train, x_val, y_train, y_val, 'rgs')

    # Test for an unsupported task_type
    with pytest.raises(ValueError, match='Unsupported task type: invalid_type.'):
        get_xgboost_objective_function(x_train, x_val, y_train, y_val, 'invalid_type')
