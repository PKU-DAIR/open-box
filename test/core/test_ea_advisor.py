import pytest
from unittest.mock import MagicMock, patch
from openbox.core.ea_advisor import EA_Advisor
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.history import Observation
from openbox.utils.constants import MAXINT, SUCCESS


def test_ea_advisor_initialization(configspace_tiny):
    config_space = configspace_tiny
    advisor = EA_Advisor(config_space)
    assert advisor.config_space == config_space
    assert advisor.num_objectives == 1
    assert advisor.num_constraints == 0
    assert advisor.population_size == 30
    assert advisor.subset_size == 20
    assert advisor.epsilon == 0.2
    assert advisor.strategy == 'worst'
    assert advisor.optimization_strategy == 'ea'
    assert advisor.batch_size == 1
    assert advisor.output_dir == 'logs'
    assert advisor.task_id == 'OpenBox'
    assert advisor.rng is not None
    assert advisor.config_space_seed is not None
    assert advisor.init_num == 1
    assert advisor.running_configs == []
    assert advisor.all_configs == set()
    assert advisor.age == 0
    assert advisor.population == []
    assert advisor.history is not None

    config = advisor.get_suggestion()
    assert config in advisor.all_configs
    assert config in advisor.running_configs

    configs = advisor.get_suggestions(batch_size=5)
    assert len(configs) == 5
    for config in configs:
        assert config in advisor.all_configs
        assert config in advisor.running_configs

    # test get_suggestion()、update_observation()
    config = advisor.get_suggestion()
    observation = Observation(config, [0.1], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
    advisor.update_observation(observation)
    assert config not in advisor.running_configs
    assert len(advisor.population) == 1

    # test sample_random_config()
    config = advisor.get_suggestions(batch_size=1)[0]
    assert config is not None

    # test get_history()
    history = advisor.get_history()
    assert history == advisor.history
