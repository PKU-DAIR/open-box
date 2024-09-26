import pytest
import numpy as np
from openbox.core.ea.differential_ea_advisor import DifferentialEAAdvisor
from openbox.utils.config_space import ConfigurationSpace, Configuration
from openbox.utils.history import Observation
from openbox.utils.constants import MAXINT, SUCCESS


def test_cmaes_ea_advisor_initialization(configspace_tiny, configspace_cat):
    config_space = configspace_tiny
    advisor = DifferentialEAAdvisor(config_space, num_objectives=2, population_size=4)
    config_cat = configspace_cat
    #添加变化的f以及cr

    '''
    在单目标优化中，允许使用动态的 f 和 cr 来提升算法的表现。
    在多目标优化中，为了保持多个目标的平衡，限制动态调整 f 和 cr，因此在这种情况下需要固定这些参数。
    断言的目的是避免算法在不合适的情况下使用动态参数，确保差分进化算法在不同优化问题上的一致性和正确性。
    '''
    advisor_vary = DifferentialEAAdvisor(config_space, f=(0,1), cr=(0,1), num_objectives=1, population_size=4)
    advisor_cat = DifferentialEAAdvisor(config_cat, num_objectives=1, population_size=4)
    assert advisor.config_space == config_space
    assert advisor.num_objectives == 2
    assert advisor.num_constraints == 0
    assert advisor.population_size == 4
    assert advisor.optimization_strategy == 'ea'
    assert advisor.constraint_strategy == 'discard'
    assert advisor.batch_size == 1
    assert advisor.output_dir == 'logs'
    assert advisor.required_evaluation_count == advisor.population_size
    assert advisor.auto_step is True
    assert advisor.strict_auto_step is True
    assert advisor.skip_gen_population is False
    assert advisor.filter_gen_population is None
    assert advisor.keep_unexpected_population is True
    assert advisor.save_cached_configuration is True
    perfs = [[2, 1], [1, 2], [2, 2], [1, 1], [1, 2], [1, 1], [1, 1], [-1, 1]]
    for i in range(8):
        suggestion1 = advisor.get_suggestion()
        assert isinstance(suggestion1, Configuration)
        observation1 = Observation(suggestion1, perfs[i], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        #这个update内有next_population的更新，会把suggestion1也就是gen的config加入sub，让sel能使用sub
        advisor.update_observation(observation1)
    
        suggestion2 = advisor_vary.get_suggestion()
        assert isinstance(suggestion2, Configuration)
        observation2 = Observation(suggestion2, [perfs[i][0]], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        advisor_vary.update_observation(observation2)

    # suggestion3 = advisor_cat.get_suggestion()
    # assert isinstance(suggestion3, Configuration)
    # observation3 = Observation(suggestion3, [1], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
    # advisor_cat.update_observation(observation3)

    config_a = config_space.sample_configuration()
    config_b = config_space.sample_configuration()
    config_c = config_space.sample_configuration()
    config = advisor.mutate(config_a, config_b, config_c, 0.5)
    assert isinstance(config, Configuration)

    config = advisor.cross_over(config_a, config_b, 0.5)
    assert isinstance(config, Configuration)


    config_a = config_cat.sample_configuration()
    config_b = config_cat.sample_configuration()
    config_c = config_cat.sample_configuration()
    config = advisor_cat.mutate(config_a, config_b, config_c, 0.5)
        #这里的prefs对应ref_point吧？传到Observation中对应objectives属性