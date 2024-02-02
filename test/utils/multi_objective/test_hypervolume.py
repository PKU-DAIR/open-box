import pytest
import numpy as np
from openbox.utils.multi_objective.hypervolume import Hypervolume, Node, MultiList, sort_by_dimension


@pytest.fixture
def sample_data_rp():
    # 提供一些样本数据用于测试
    ref_point = np.array([6, 7])
    pareto_Y = np.array([[1, 3], [3, 4], [2, 5], [5, 3], [4, 2]])
    return ref_point, pareto_Y


def test_initialization(sample_data_rp):
    ref_point, pareto_Y = sample_data_rp
    hypervolume = Hypervolume(ref_point=ref_point)
    assert np.array_equal(hypervolume.ref_point, ref_point)


def test_compute_hypervolume(sample_data_rp):
    ref_point, pareto_Y = sample_data_rp
    hypervolume = Hypervolume(ref_point=ref_point)
    result = hypervolume.compute(pareto_Y=pareto_Y)
    assert result >= 0


def test_multilist_operations(sample_data_rp):
    ref_point, pareto_Y = sample_data_rp
    hypervolume = Hypervolume(ref_point=ref_point)

    # 初始化 MultiList
    hypervolume._initialize_multilist(pareto_Y)
    assert hypervolume.list is not None


@pytest.fixture
def sample_data_list():
    # Provide some sample data for testing
    m = 2  # Number of objectives
    node1 = Node(m=m, data=np.array([1, 2]))
    node2 = Node(m=m, data=np.array([3, 4]))
    node3 = Node(m=m, data=np.array([2, 5]))
    return m, node1, node2, node3


def test_multi_list_initialization(sample_data_list):
    m, _, _, _ = sample_data_list
    multi_list = MultiList(m=m)
    assert multi_list.m == m
    assert multi_list.sentinel.next == [multi_list.sentinel] * m
    assert multi_list.sentinel.prev == [multi_list.sentinel] * m


def test_multi_list_append(sample_data_list):
    m, node1, _, _ = sample_data_list
    multi_list = MultiList(m=m)

    # Test appending a node to the end of the list at index 0
    multi_list.append(node1, index=0)
    assert multi_list.sentinel.prev[0] == node1
    assert node1.next[0] == multi_list.sentinel


def test_multi_list_extend(sample_data_list):
    m, node1, node2, _ = sample_data_list
    multi_list = MultiList(m=m)

    # Test extending the list at index 0 with nodes node1 and node2
    multi_list.extend([node1, node2], index=0)
    assert multi_list.sentinel.prev[0] == node2
    assert node2.next[0] == multi_list.sentinel
    assert node1.next[0] == node2
    assert node2.prev[0] == node1


def test_multi_list_remove(sample_data_list):
    m, node1, _, _ = sample_data_list
    multi_list = MultiList(m=m)

    # Test removing a node from all lists in [0, m-1]
    bounds = np.full((2, m), float("inf"))
    multi_list.append(node1, index=0)
    removed_node = multi_list.remove(node1, index=m - 1, bounds=bounds)
    assert removed_node == node1


def test_multi_list_reinsert(sample_data_list):
    m, node1, _, _ = sample_data_list
    multi_list = MultiList(m=m)

    # Test re-inserting a node at its original position
    bounds = np.full((2, m), float("inf"))
    multi_list.append(node1, index=0)
    removed_node = multi_list.remove(node1, index=m - 1, bounds=bounds)
    multi_list.reinsert(removed_node, index=m - 1, bounds=bounds)
    assert node1.next[0] == multi_list.sentinel


@pytest.fixture
def sample_nodes():
    # Provide some sample nodes for testing
    m = 3  # Number of objectives
    node1 = Node(m=m, data=np.array([1, 1, 3]))
    node2 = Node(m=m, data=np.array([3, 2, 2]))
    node3 = Node(m=m, data=np.array([2, 3, 1]))
    return m, [node1, node2, node3]


def test_sort_by_dimension(sample_nodes):
    _, nodes = sample_nodes

    # Test sorting nodes by the first dimension
    sort_by_dimension(nodes, i=0)
    assert nodes[0].data.tolist() == [1, 1, 3]
    assert nodes[1].data.tolist() == [2, 3, 1]
    assert nodes[2].data.tolist() == [3, 2, 2]

    # Test sorting nodes by the second dimension
    sort_by_dimension(nodes, i=1)
    assert nodes[0].data.tolist() == [1, 1, 3]
    assert nodes[1].data.tolist() == [3, 2, 2]
    assert nodes[2].data.tolist() == [2, 3, 1]

    # Test sorting nodes by the third dimension
    sort_by_dimension(nodes, i=2)
    assert nodes[0].data.tolist() == [2, 3, 1]
    assert nodes[1].data.tolist() == [3, 2, 2]
    assert nodes[2].data.tolist() == [1, 1, 3]