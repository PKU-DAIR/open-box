import pytest
import numpy as np
from openbox.core.ea.differential_ea_advisor import DifferentialEAAdvisor
from openbox.utils.config_space import ConfigurationSpace, Configuration
from openbox.utils.history import Observation
from openbox.utils.constants import MAXINT, SUCCESS


def test_cmaes_ea_advisor_initialization(configspace_tiny):
    config_space = configspace_tiny
    advisor = DifferentialEAAdvisor(config_space, num_objectives=2, population_size=4)
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

    perfs = [[2, 1], [1, 2], [2, 2], [1, 1], [1, 2]]
    for i in range(5):
        suggestion1 = advisor.get_suggestion()
        assert isinstance(suggestion1, Configuration)
        observation1 = Observation(suggestion1, perfs[i], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        advisor.update_observation(observation1)

    config_a = config_space.sample_configuration()
    config_b = config_space.sample_configuration()
    config_c = config_space.sample_configuration()
    config = advisor.mutate(config_a, config_b, config_c, 0.5)
    assert isinstance(config, Configuration)

    config = advisor.cross_over(config_a, config_b, 0.5)
    assert isinstance(config, Configuration)
