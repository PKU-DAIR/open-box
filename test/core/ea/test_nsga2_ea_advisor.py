import pytest
from unittest.mock import MagicMock, patch
from openbox.core.ea.nsga2_ea_advisor import NSGA2EAdvisor
from openbox.utils.config_space import ConfigurationSpace, Configuration
from openbox.utils.history import Observation
from openbox.utils.constants import MAXINT, SUCCESS


def test_nsga2_ea_advisor_initialization(configspace_tiny):
    config_space = configspace_tiny
    advisor = NSGA2EAdvisor(config_space, num_objectives=2, subset_size=2, population_size=4)
    assert advisor.config_space == config_space
    assert advisor.num_objectives == 2
    assert advisor.num_constraints == 0
    assert advisor.population_size == 4
    assert advisor.optimization_strategy == 'ea'
    assert advisor.batch_size == 1
    assert advisor.output_dir == 'logs'
    assert advisor.subset_size == 2
    assert advisor.epsilon == 0.2
    assert advisor.strategy == 'worst'

    perfs = [[2, 1], [1, 2], [2, 2], [1, 1], [1, 2]]
    for i in range(5):
        suggestion1 = advisor.get_suggestion()
        assert isinstance(suggestion1, Configuration)
        observation1 = Observation(suggestion1, perfs[i], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        advisor.update_observation(observation1)