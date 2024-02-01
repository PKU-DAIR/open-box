import pytest
import numpy as np
from openbox.utils.multi_objective.pareto import is_non_dominated, get_pareto_front


@pytest.fixture
def sample_data_dominated():
    # Provide some sample data for testing
    dominated_points = np.array([[1, 6], [2, 4], [3, 5]])

    return dominated_points


def test_is_non_dominated(sample_data_dominated):
    dominated_points = sample_data_dominated

    # Test dominated points
    result_dominated = is_non_dominated(dominated_points)
    assert result_dominated[0] & result_dominated[1] & ~result_dominated[2]


def test_get_pareto_front(sample_data_dominated):
    dominated_points = sample_data_dominated

    # Test dominated points
    pareto_front = get_pareto_front(dominated_points)
    assert np.array_equal(pareto_front, np.array([[1, 6], [2, 4]]))
