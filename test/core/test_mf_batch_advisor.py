import pytest
from unittest.mock import MagicMock, patch
from openbox.core.mf_batch_advisor import MFBatchAdvisor
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.history import Observation
from openbox.utils.constants import MAXINT, SUCCESS


def test_mf_batch_advisor(configspace_tiny, history_single_obs):
    config_space = configspace_tiny
    advisor = MFBatchAdvisor(config_space)
    assert advisor.config_space == config_space
    assert advisor.num_objectives == 1
    assert advisor.num_constraints == 0
    assert advisor.batch_size == 4
    assert advisor.init_num == 3
    assert advisor.init_strategy == 'random_explore_first'
    assert advisor.rand_prob == 0.1
    assert advisor.optimization_strategy == 'bo'
    assert advisor.surrogate_type == 'mfgpe'
    assert advisor.acq_type == 'ei'
    assert advisor.acq_optimizer_type == 'local_random'
    assert advisor.ref_point is None
    assert advisor.output_dir == 'logs'
    assert advisor.task_id == 'OpenBox'
    assert advisor.rng is not None
    assert advisor.history_list == []
    assert advisor.resource_identifiers == []

    resource_ratios = [0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.8, 0.8, 1.0, 1.0]

    for i, obs in enumerate(history_single_obs.observations):
        advisor.update_observation(obs, resource_ratio=resource_ratios[i])

    suggestions = advisor.get_suggestions(batch_size=5, history=history_single_obs)
    assert len(suggestions) == 5

    observation = Observation(suggestions[0], [0.1], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
    advisor.update_observation(observation, resource_ratio=1)
    assert len(advisor.history) == 3

    configs = advisor.sample_random_configs(config_space, 5)
    assert len(configs) == 5

    assert len(advisor.history_list) == 4
    assert len(advisor.resource_identifiers) == 4
