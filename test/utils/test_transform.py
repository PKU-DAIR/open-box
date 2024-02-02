import pytest
import numpy as np
from openbox.utils.transform import zero_one_normalization, zero_one_unnormalization, zero_mean_unit_var_normalization, \
    zero_mean_unit_var_unnormalization, bilog_transform, gaussian_transform, get_transform_function


def test_zero_one_normalization_with_valid_input():
    X = np.array([1, 2, 3, 4, 5])
    X_normalized, lower, upper = zero_one_normalization(X)
    assert np.allclose(X_normalized, np.array([0., 0.25, 0.5, 0.75, 1.]))
    assert lower == 1
    assert upper == 5


def test_zero_one_unnormalization_with_valid_input():
    X_normalized = np.array([0., 0.25, 0.5, 0.75, 1.])
    X = zero_one_unnormalization(X_normalized, 1, 5)
    assert np.allclose(X, np.array([1, 2, 3, 4, 5]))


def test_zero_mean_unit_var_normalization_with_valid_input():
    X = np.array([1, 2, 3, 4, 5])
    X_normalized, mean, std = zero_mean_unit_var_normalization(X)
    assert np.allclose(X_normalized, np.array([-1.41411357, -0.70705678, 0., 0.70705678, 1.41411357]))
    assert mean == 3
    assert np.allclose(std, 1.41431356)


def test_zero_mean_unit_var_unnormalization_with_valid_input():
    X_normalized = np.array([-1.41421356, -0.70710678, 0., 0.70710678, 1.41421356])
    X = zero_mean_unit_var_unnormalization(X_normalized, 3, 1.41421356)
    assert np.allclose(X, np.array([1, 2, 3, 4, 5]))


def test_bilog_transform_with_valid_input():
    X = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    X_transformed = bilog_transform(X)
    assert np.allclose(X_transformed, np.array([0.09531018, 0.18232156, 0.26236426, 0.33647224, 0.40546511]))


def test_gaussian_transform_with_valid_input():
    X = np.array([1, 2, 3, 4, 5])
    X_transformed = gaussian_transform(X)
    assert np.allclose(X_transformed, np.array([-1.44413311, -0.67448975,  0.,  0.67448975,  1.44413311]))


def test_get_transform_function_with_valid_input():
    transform_func = get_transform_function('bilog')
    assert transform_func == bilog_transform


def test_get_transform_function_with_invalid_input():
    with pytest.raises(ValueError):
        get_transform_function('invalid_transform')
