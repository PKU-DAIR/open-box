import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from openbox.core.generic_advisor import Advisor
from openbox.utils.util_funcs import check_random_state
from openbox.utils.history import Observation
from openbox.utils.constants import MAXINT, SUCCESS


def test_generic_advisor(configspace_tiny, history_single_obs):
    config_space = configspace_tiny
    advisor = Advisor(config_space)
    assert advisor.config_space == config_space
    assert advisor.num_objectives == 1
    assert advisor.num_constraints == 0
    assert advisor.init_strategy == 'random_explore_first'
    assert advisor.rand_prob == 0.1
    assert advisor.optimization_strategy == 'bo'
    assert advisor.surrogate_type == 'gp'
    assert advisor.acq_type == 'ei'
    assert advisor.acq_optimizer_type == 'random_scipy'
    assert advisor.ref_point is None
    assert advisor.output_dir == 'logs'
    assert advisor.task_id == 'OpenBox'
    assert isinstance(advisor.rng, np.random.RandomState)

    config = advisor.get_suggestion(history=history_single_obs)
    assert config is not None

    observation = Observation(config, [0.1], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
    advisor.update_observation(observation)
    assert len(advisor.history) == 1

    configs = advisor.sample_random_configs(config_space, 5)
    assert len(configs) == 5

    history = advisor.get_history()
    assert history == advisor.history

    advisor.save_json("test/datas/test.json")

    advisor.load_json("test/datas/test.json")

