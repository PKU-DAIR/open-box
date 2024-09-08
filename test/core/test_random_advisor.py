import pytest
from unittest.mock import MagicMock, patch
from openbox.core.random_advisor import RandomAdvisor
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.history import History, Observation
from openbox.utils.constants import MAXINT, SUCCESS
from openbox.utils.early_stop import EarlyStopException

def test_random_advisor_initialization(configspace_tiny, history_single_obs):
    config_space = configspace_tiny
    advisor = RandomAdvisor(config_space)
    assert advisor.config_space == config_space
    assert advisor.num_objectives == 1
    assert advisor.num_constraints == 0
    assert advisor.ref_point is None
    assert advisor.output_dir == 'logs'
    assert advisor.task_id == 'OpenBox'
    assert advisor.rng is not None

    config = advisor.get_suggestion(history=history_single_obs)
    assert config is not None

    observation = Observation(config, [0.1], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
    advisor.update_observation(observation)
    assert len(advisor.history) == 1

    configs = advisor.sample_random_configs(config_space, 5)
    assert len(configs) == 5


def test_random_advisor_early_stop(configspace_tiny, history_single_obs):
    config_space = configspace_tiny
    advisor = RandomAdvisor(config_space, early_stop=True, early_stop_kwargs={'max_no_improvement_rounds': 10})

    for i in range(13):
        if i < 12:
            config = advisor.get_suggestion()
            observation = Observation(config, [0.1], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
            advisor.update_observation(observation)
            assert len(advisor.history) == i + 1
        else:
            with pytest.raises(EarlyStopException):
                config = advisor.get_suggestion()

