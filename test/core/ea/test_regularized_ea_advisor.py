import pytest
from unittest.mock import MagicMock, patch
from openbox.core.ea.regularized_ea_advisor import RegularizedEAAdvisor
from ConfigSpace import ConfigurationSpace, Configuration
from openbox.utils.history import Observation
from openbox.utils.constants import MAXINT, SUCCESS


def test_regularized_ea_advisor(configspace_tiny):
    config_space = configspace_tiny
    advisor = RegularizedEAAdvisor(config_space, population_size=4, subset_size=2, num_objectives=2)
    assert advisor.config_space == config_space
    assert advisor.subset_size == 2
    assert advisor.epsilon == 0.2
    assert advisor.strategy == 'worst'

    perfs = [[2, 1], [1, 2], [2, 2], [1, 1], [1, 2]]
    for i in range(5):
        suggestion1 = advisor.get_suggestion()
        assert isinstance(suggestion1, Configuration)
        observation1 = Observation(suggestion1, perfs[i], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        advisor.update_observation(observation1)