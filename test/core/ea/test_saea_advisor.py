import pytest
from unittest.mock import MagicMock, patch
from openbox.core.ea.saea_advisor import SAEAAdvisor
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.history import Observation
from ConfigSpace import ConfigurationSpace, Configuration
from openbox.utils.history import Observation
from openbox.utils.constants import MAXINT, SUCCESS


def test_saea_advisor(configspace_tiny):
    config_space = configspace_tiny
    advisor = SAEAAdvisor(config_space, population_size=4, num_objectives=1)
    assert advisor.config_space == config_space
    assert advisor.gen_multiplier == 50
    assert advisor.is_models_trained is False

    perfs = [[2, 1], [1, 2], [2, 2], [1, 1], [1, 2]]
    for i in range(5):
        suggestion1 = advisor.get_suggestion()
        assert isinstance(suggestion1, Configuration)
        observation1 = Observation(suggestion1, perfs[i][:1], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        advisor.update_observation(observation1)
