import pytest
from openbox.optimizer.nsga_base import NSGABase
from openbox.utils.config_space import ConfigurationSpace


def test_nsga_base_initialization(configspace_tiny, func_brain):
    config_space = configspace_tiny
    objective_function = func_brain
    nsga_base = NSGABase(objective_function, config_space)
    assert nsga_base.config_space == config_space
    assert nsga_base.objective_function == objective_function
    assert nsga_base.task_id == 'OpenBox'
    assert nsga_base.output_dir == 'logs/'
    assert nsga_base.rng is not None
    assert nsga_base.max_runs == 2500
    assert nsga_base.rng is not None

    with pytest.raises(NotImplementedError):
        nsga_base.run()

    with pytest.raises(NotImplementedError):
        nsga_base.iterate()

    with pytest.raises(NotImplementedError):
        nsga_base.get_incumbents()