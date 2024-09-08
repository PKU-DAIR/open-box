import pytest
from unittest.mock import MagicMock, patch

from openbox.utils.early_stop import EarlyStopException
from openbox.core.async_batch_advisor import AsyncBatchAdvisor
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.history import History, Observation
from openbox.utils.constants import MAXINT, SUCCESS
from openbox import logger


def test_async_batch_advisor_initialization(configspace_tiny, history_single_obs):
    config_space = configspace_tiny
    advisor = AsyncBatchAdvisor(config_space)
    assert advisor.config_space == config_space
    assert advisor.batch_size == 4
    assert advisor.batch_strategy == 'default'
    assert advisor.running_configs == []
    assert advisor.bo_start_n == 3
    assert advisor.acq_type == 'ei'

    suggestion = advisor.get_suggestion(history=history_single_obs)
    assert suggestion in advisor.running_configs

    advisor.batch_strategy = 'median_imputation'
    suggestion = advisor.get_suggestion(history=history_single_obs)
    assert suggestion in advisor.running_configs

    advisor.batch_strategy = 'local_penalization'
    suggestion = advisor.get_suggestion(history=history_single_obs)
    assert suggestion in advisor.running_configs

    observation = Observation(suggestion, [0.1], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
    advisor.update_observation(observation)
    assert len(advisor.history) == 1

