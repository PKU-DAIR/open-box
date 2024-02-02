import pytest
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.tuning.config_space import get_config_space, get_lightgbm_config_space, get_xgboost_config_space


def test_get_config_space():
    # Test for a supported model (lightgbm)
    lightgbm_config_space = get_config_space('lightgbm')
    assert isinstance(lightgbm_config_space, ConfigurationSpace)

    # Test for another supported model (xgboost)
    xgboost_config_space = get_config_space('xgboost')
    assert isinstance(xgboost_config_space, ConfigurationSpace)

    # Test for an unsupported model
    with pytest.raises(ValueError, match='Unsupported model: invalid_model.'):
        get_config_space('invalid_model')


def test_get_lightgbm_config_space():
    # Test for task_type 'cls'
    cs_cls = get_lightgbm_config_space('cls')
    assert isinstance(cs_cls, ConfigurationSpace)

    # Test for task_type 'rgs'
    with pytest.raises(NotImplementedError):
        get_lightgbm_config_space('rgs')

    # Test for an unsupported task_type
    with pytest.raises(ValueError, match='Unsupported task type: invalid_type.'):
        get_lightgbm_config_space('invalid_type')


def test_get_xgboost_config_space():
    # Test for task_type 'cls'
    cs_cls = get_xgboost_config_space('cls')
    assert isinstance(cs_cls, ConfigurationSpace)

    # Test for task_type 'rgs'
    with pytest.raises(NotImplementedError):
        get_xgboost_config_space('rgs')

    # Test for an unsupported task_type
    with pytest.raises(ValueError, match='Unsupported task type: invalid_type.'):
        get_xgboost_config_space('invalid_type')