import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from openbox.core.tpe_advisor import TPE_Advisor
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.history import History, Observation
from openbox.utils.constants import MAXINT, SUCCESS


def test_tpe_advisor_initialization(configspace_tiny, history_single_obs):
    config_space = configspace_tiny
    advisor = TPE_Advisor(config_space)
    assert advisor.config_space == config_space
    assert advisor.top_n_percent == 15
    assert advisor.num_samples == 64
    assert advisor.random_fraction == 1 / 3
    assert advisor.bw_factor == 3
    assert advisor.min_bandwidth == 1e-3
    assert advisor.task_id == 'OpenBox'
    assert advisor.output_dir == 'logs'
    assert advisor.rng is not None

    config = advisor.get_suggestion(history=history_single_obs)
    observation = Observation(config, [0.1], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
    advisor.update_observation(observation)
    assert len(advisor.history) == 1

    # test impute_conditional_data
    array = np.array([[1, 2, np.nan], [4, np.nan, 6], [3, 8, 9]])
    imputed_array = advisor.impute_conditional_data(array)
    assert not np.isnan(imputed_array).any()

    advisor.fit_kde_models(history_single_obs)
    assert 'good' in advisor.kde_models
    assert 'bad' in advisor.kde_models

    configs = advisor.sample_random_configs(config_space, 5)
    assert len(configs) == 5

    history = advisor.get_history()
    assert history == advisor.history
    