import numpy as np
import pytest
from unittest.mock import MagicMock
from openbox.core.pso.base_pso_advisor import BasePSOAdvisor, Individual, pareto_sort, pareto_frontier, pareto_best, pareto_layers, constraint_check
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.history import Observation
from openbox.utils.constants import MAXINT, SUCCESS


def test_base_pso_advisor(configspace_tiny):
    config_space = configspace_tiny
    advisor = BasePSOAdvisor(config_space)
    assert advisor.config_space == config_space
    assert advisor.num_objectives == 1
    assert advisor.num_constraints == 0
    assert advisor.population_size == 30
    assert advisor.batch_size == 1
    assert advisor.output_dir == 'logs'
    assert advisor.all_configs == set()
    assert advisor.population == []
    assert advisor.history is not None

    with pytest.raises(NotImplementedError):
        advisor.get_suggestions()

    config = advisor.sample_random_config(config_space)
    observation = Observation(config, [0.1], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})

    with pytest.raises(NotImplementedError):
        advisor.update_observations(observation)

    assert advisor.get_history() is not None


def test_individual_initialization_pso():
    pos = np.array([0.1, 0.2, 0.3])
    vel = np.array([1.0, 2.0, 3.0])
    perf = 0.5
    constraints_satisfied = True
    data = {'key': 'value'}
    individual = Individual(pos, vel, perf, constraints_satisfied, data)
    assert np.all(individual.pos == pos)
    assert np.all(individual.vel == vel)
    assert individual.perf == perf
    assert individual.constraints_satisfied == constraints_satisfied
    assert individual.data == data

    assert individual.perf_1d() == perf

    assert np.all(individual['pos'] == pos)
    assert np.all(individual['vel'] == vel)
    assert individual['perf'] == perf
    assert individual['constraints_satisfied'] == constraints_satisfied
    assert individual['key'] == 'value'


def test_pareto_sort_pso(configspace_tiny):
    population = []
    perfs = [3, 2, 5, 1, 4]
    for i in range(5):
        population.append(Individual(np.zeros(1), np.zeros(1), [perfs[i]], allow_constraint=True))

    sorted_population = pareto_sort(population)
    assert len(sorted_population) == len(population)

    sorted_population = pareto_sort(population, ascending=True)
    assert len(sorted_population) == len(population)


def test_pareto_frontier_with_single_dimension_pso(configspace_tiny):
    population = []
    perfs = [3, 2, 5, 1, 4]
    for i in range(5):
        population.append(Individual(np.zeros(1), np.zeros(1), [perfs[i]], allow_constraint=True))

    sorted_population = pareto_frontier(population)
    assert len(sorted_population) == 1
    assert sorted_population[0].perf == [1.0]


def test_pareto_frontier_with_multiple_dimensions_pso(configspace_tiny):
    population = []
    perfs = [[3, 1], [2, 2], [5, 1], [1, 3], [4, 5]]
    for i in range(5):
        population.append(Individual(np.zeros(1), np.zeros(1), perfs[i], allow_constraint=True))

    sorted_population = pareto_frontier(population)
    assert len(sorted_population) == 4


def test_pareto_best_with_single_dimension_pso(configspace_tiny):
    population = []
    perfs = [3, 2, 5, 1, 4]
    for i in range(5):
        population.append(Individual(np.zeros(1), np.zeros(1), [perfs[i]], allow_constraint=True))

    sorted_population = pareto_best(population, count=3)
    assert len(sorted_population) == 3
    for individual in sorted_population:
        assert individual in population


def test_pareto_best_with_multiple_dimensions_pso(configspace_tiny):
    population = []
    config_space = configspace_tiny
    perfs = [[3, 1], [2, 2], [5, 1], [1, 3], [4, 5]]
    for i in range(5):
        population.append(Individual(np.zeros(1), np.zeros(1), perfs[i], allow_constraint=True))

    sorted_population = pareto_best(population, count=3)
    assert len(sorted_population) == 3
    for individual in sorted_population:
        assert individual in population

    sorted_population = pareto_best(population, count_ratio=0.6)
    assert len(sorted_population) == 3
    for individual in sorted_population:
        assert individual in population


def test_pareto_layers_pso(configspace_tiny):
    population = []
    perfs = [[3, 1], [2, 2], [5, 1], [1, 3], [4, 5]]
    for i in range(5):
        population.append(Individual(np.zeros(1), np.zeros(1), perfs[i], allow_constraint=True))

    layers = pareto_layers(population)
    assert len(layers) == 2
    assert len(layers[0]) == 4
    assert layers[1][0] == population[4]


def test_constraint_check_pso():
    assert constraint_check(None) is True

    assert constraint_check(True) is True
    assert constraint_check(False) is False

    assert constraint_check(0.5) is False
    assert constraint_check(-0.5) is True
    assert constraint_check(0.5, positive_numbers=True) is True
    assert constraint_check(-0.5, positive_numbers=True) is False

    assert constraint_check(1) is False
    assert constraint_check(-1) is True
    assert constraint_check(1, positive_numbers=True) is True
    assert constraint_check(-1, positive_numbers=True) is False

    assert constraint_check([True, 0.5, 1]) is False
    assert constraint_check([True, -0.5, -1]) is True
    assert constraint_check([True, 0.5, 1], positive_numbers=True) is True
    assert constraint_check([True, -0.5, -1], positive_numbers=True) is False