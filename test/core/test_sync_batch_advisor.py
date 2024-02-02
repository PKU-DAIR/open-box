
import pytest
from unittest.mock import MagicMock, patch
from openbox.core.sync_batch_advisor import SyncBatchAdvisor
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.history import History, Observation
from openbox.utils.constants import MAXINT, SUCCESS


def test_sync_batch_advisora(configspace_tiny, history_single_obs):
    config_space = configspace_tiny
    advisor = SyncBatchAdvisor(config_space)
    assert advisor.config_space == config_space
    assert advisor.num_objectives == 1
    assert advisor.num_constraints == 0
    assert advisor.batch_size == 4
    assert advisor.batch_strategy == 'default'
    assert advisor.init_num == 3
    assert advisor.init_strategy == 'random_explore_first'
    assert advisor.optimization_strategy == 'bo'
    assert advisor.surrogate_type == 'gp'
    assert advisor.acq_type == 'ei'
    assert advisor.acq_optimizer_type == 'random_scipy'
    assert advisor.ref_point is None
    assert advisor.output_dir == 'logs'
    assert advisor.task_id == 'OpenBox'
    assert advisor.rng is not None

    suggestions = advisor.get_suggestions(batch_size=5, history=history_single_obs)
    assert len(suggestions) == 5

    suggestions = advisor.get_suggestions(batch_size=5)
    assert len(suggestions) == 5

    suggestions = advisor.get_suggestions(history=history_single_obs)
    assert len(suggestions) == advisor.batch_size

    advisor.batch_strategy = 'median_imputation'
    assert len(advisor.get_suggestions(history=history_single_obs)) == advisor.batch_size

    advisor.batch_strategy = 'local_penalization'
    assert len(advisor.get_suggestions(history=history_single_obs)) == advisor.batch_size

    advisor.batch_strategy = 'reoptimization'
    assert len(advisor.get_suggestions(history=history_single_obs)) == advisor.batch_size

    advisor.batch_strategy = 'invalid strategy'
    with pytest.raises(ValueError):
        advisor.get_suggestions(history=history_single_obs)

    observation = Observation(suggestions[0], [0.1], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
    advisor.update_observation(observation)
    assert len(advisor.history) == 1
