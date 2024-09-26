import pytest
from openbox.core.ea.adaptive_ea_advisor import AdaptiveEAAdvisor
from openbox.utils.config_space import ConfigurationSpace, Configuration
from openbox.utils.history import Observation
from openbox.utils.constants import MAXINT, SUCCESS


def test_adaptive_ea_advisor(configspace_tiny):
    config_space = configspace_tiny
    advisor = AdaptiveEAAdvisor(config_space, population_size=4, subset_size=2, num_objectives=2, pc=1, pm=1)
    assert advisor.config_space == config_space
    assert advisor.subset_size == 2
    assert advisor.epsilon == 0.2
    assert advisor.pm == 1
    assert advisor.pc == 1
    assert advisor.strategy == 'worst'
    assert advisor.k1 == 0.25
    assert advisor.k2 == 0.3
    assert advisor.k3 == 0.25
    assert advisor.k4 == 0.3
    assert advisor.last_suggestions == []
    assert advisor.last_observations == []

    perfs = [[2, 1], [1, 2], [2, 2], [1, 1], [1, 2]]
    for i in range(4):
        suggestion1 = advisor.get_suggestion()
        assert isinstance(suggestion1, Configuration)
        observation1 = Observation(suggestion1, perfs[i], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        advisor.update_observation(observation1)

    config_a = config_space.sample_configuration()
    config_b = config_space.sample_configuration()
    next_config = advisor.cross_over(config_a, config_b)
    assert isinstance(next_config, Configuration)

    config = config_space.sample_configuration()
    next_config = advisor.mutation(config)
    assert isinstance(next_config, Configuration)