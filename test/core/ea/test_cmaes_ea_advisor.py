import pytest
import numpy as np
from openbox.core.ea.cmaes_ea_advisor import CMAESEAAdvisor
from openbox.utils.config_space import ConfigurationSpace, Configuration
from openbox.utils.history import Observation
from openbox.utils.constants import MAXINT, SUCCESS


def test_cmaes_ea_advisor_initialization(configspace_tiny):
    config_space = configspace_tiny
    advisor = CMAESEAAdvisor(config_space, num_objectives=2, population_size=4)
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
    assert advisor.mu == int(advisor.lam / 2)
    assert advisor.w is not None
    assert advisor.cs is not None
    assert advisor.ds is not None
    assert advisor.cc is not None
    assert advisor.mu_cov == advisor.mu_eff
    assert advisor.c_cov is not None
    assert advisor.ps is not None
    assert advisor.pc is not None
    assert advisor.cov is not None
    assert advisor.mean is not None
    assert advisor.sigma == 0.5
    assert advisor.generation_id == 0
    assert advisor.unvalidated_map == dict()

    perfs = [[2, 1], [1, 2], [2, 2], [1, 1], [1, 2]]
    for i in range(5):
        suggestion1 = advisor.get_suggestion()
        assert isinstance(suggestion1, Configuration)
        observation1 = Observation(suggestion1, perfs[i], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        advisor.update_observation(observation1)
