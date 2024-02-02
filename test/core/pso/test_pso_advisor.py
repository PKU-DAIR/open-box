import pytest
from unittest.mock import MagicMock, patch
from openbox.core.pso.pso_advisor import PSOAdvisor
from ConfigSpace import ConfigurationSpace, Configuration
from openbox.utils.history import Observation
from openbox.utils.constants import MAXINT, SUCCESS


def test_pso_advisor_initialization(configspace_tiny):
    config_space = configspace_tiny
    advisor = PSOAdvisor(config_space, num_objectives=2, population_size=4)
    assert advisor.config_space == config_space
    assert advisor.max_iter == None
    assert advisor.cur_iter == 0
    assert advisor.w_stg == 'default'
    assert advisor.det == 0.999
    assert advisor.wi == 0.729
    assert advisor.c1 == 1.3
    assert advisor.c2 == 1.3
    assert advisor.pbest == []

    configs = advisor.get_suggestions()
    assert len(configs) == 4
    assert len(advisor.running_configs) == 4
    assert len(advisor.all_configs) == 4

    perfs = [[2, 1], [1, 2], [2, 2], [1, 1]]
    observations = []
    for i in range(4):
        suggestion1 = configs[i]
        assert isinstance(suggestion1, Configuration)
        observation1 = Observation(suggestion1, perfs[i], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        observations.append(observation1)
    advisor.update_observations(observations)
    assert len(advisor.pbest) == 4