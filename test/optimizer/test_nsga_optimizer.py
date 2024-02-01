import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from openbox.optimizer.nsga_optimizer import NSGAOptimizer


def test_nsga_optimizer(configspace_tiny, func_brain):
    config_space = configspace_tiny
    objective_function = func_brain
    nsga_optimizer = NSGAOptimizer(objective_function, config_space, max_runs=10)
    assert nsga_optimizer.config_space == config_space
    assert nsga_optimizer.objective_function == objective_function
    assert nsga_optimizer.problem is not None
    assert nsga_optimizer.algorithm is not None

    nsga_optimizer.run()

    pareto_set, pareto_front = nsga_optimizer.get_incumbents()
    assert isinstance(pareto_set, list)
    assert len(pareto_set) == len(pareto_front)
    assert isinstance(pareto_front, np.ndarray)

    solutions = nsga_optimizer.get_solutions()
    assert isinstance(solutions, list)

