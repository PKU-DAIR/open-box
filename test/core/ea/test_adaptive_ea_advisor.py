import pytest
from openbox.core.ea.adaptive_ea_advisor import AdaptiveEAAdvisor
from openbox.utils.config_space import ConfigurationSpace, Configuration
from openbox.utils.history import Observation
from openbox.utils.constants import MAXINT, SUCCESS


def test_adaptive_ea_advisor(configspace_tiny):
    config_space = configspace_tiny
    advisor = AdaptiveEAAdvisor(config_space, population_size=8, subset_size=2, num_objectives=2, pc=1, pm=1)
    #换一个oldest策略再来一次?
    '''
    目前看来这个只支持单目标优化，这里的多目标似乎是假的
    '''
    advisor_oldest = AdaptiveEAAdvisor(config_space, strategy='oldest', population_size=8, subset_size=2, num_objectives=2, pc=1, pm=1)
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
    '''
    多assert
    '''
    perfs = [[2, 1], [1, 2], [2, 2], [1, 1], [1, 2], [1, 1], [1, 1], [1, 1], [1, 1], [2, 3], [4, 5], [6, 7], [3, 6]]
    #修改为10次 ref_point多给点
    for i in range(13):
        suggestion1 = advisor.get_suggestion()
        assert isinstance(suggestion1, Configuration)
        observation1 = Observation(suggestion1, perfs[i], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        advisor.update_observation(observation1)
        '''
        oldest strategy
        '''
        suggestion1 = advisor_oldest.get_suggestion()
        assert isinstance(suggestion1, Configuration)
        observation1 = Observation(suggestion1, perfs[i], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        advisor_oldest.update_observation(observation1)

    config_a = config_space.sample_configuration()
    config_b = config_space.sample_configuration()
    next_config = advisor.cross_over(config_a, config_b)
    assert isinstance(next_config, Configuration)

    config = config_space.sample_configuration()
    next_config = advisor.mutation(config)
    #测试None
    config = None
    next_config = advisor.mutation(config)
    assert isinstance(next_config, Configuration)