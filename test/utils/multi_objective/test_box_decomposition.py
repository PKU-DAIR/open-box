import numpy as np
import pytest
from openbox.utils.multi_objective.box_decomposition import NondominatedPartitioning


@pytest.fixture
def sample_data_ndp():
    # 提供一些样本数据用于测试
    num_objectives = 2
    Y = np.array([[1, 3], [3, 4], [2, 5], [5, 3], [4, 2]])
    return num_objectives, Y


def test_initialization_ndp(sample_data_ndp):
    num_objectives, Y = sample_data_ndp
    ndp = NondominatedPartitioning(num_objectives=num_objectives, Y=Y)
    assert ndp.num_objectives == num_objectives
    assert np.array_equal(ndp.Y, Y)


def test_update_ndp(sample_data_ndp):
    num_objectives, Y = sample_data_ndp
    ndp = NondominatedPartitioning(num_objectives=num_objectives)
    ndp.update(Y=Y)
    assert np.array_equal(ndp.pareto_Y, np.array([[1, 3], [4, 2]]))


def test_binary_partition_non_dominated_space_ndp(sample_data_ndp):
    num_objectives, Y = sample_data_ndp
    ndp = NondominatedPartitioning(num_objectives=num_objectives, Y=Y)
    ndp.binary_partition_non_dominated_space()
    assert len(ndp.hypercells) > 0


def test_partition_non_dominated_space_2d_ndp(sample_data_ndp):
    num_objectives, Y = sample_data_ndp
    ndp = NondominatedPartitioning(num_objectives=num_objectives, Y=Y)
    ndp.partition_non_dominated_space_2d()
    assert len(ndp.hypercells) > 0


def test_get_hypercell_bounds_ndp(sample_data_ndp):
    num_objectives, Y = sample_data_ndp
    ndp = NondominatedPartitioning(num_objectives=num_objectives, Y=Y)
    ref_point = np.array([6, 7])  # 请根据实际情况提供一个合适的参考点
    bounds = ndp.get_hypercell_bounds(ref_point=ref_point)
    assert bounds.shape == (2, len(ndp.hypercells[0]), num_objectives)


def test_compute_hypervolume_ndp(sample_data_ndp):
    num_objectives, Y = sample_data_ndp
    ndp = NondominatedPartitioning(num_objectives=num_objectives, Y=Y)
    ref_point = np.array([0, 0])  # 请根据实际情况提供一个合适的参考点
    hypervolume = ndp.compute_hypervolume(ref_point=ref_point)
    assert hypervolume == -1