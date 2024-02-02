import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from openbox.core.mc_advisor import MCAdvisor
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.history import History, Observation
from openbox.utils.util_funcs import check_random_state
from openbox.utils.constants import MAXINT, SUCCESS


def test_mc_advisor(configspace_tiny, multi_start_history_single_obs):
    config_space = configspace_tiny
    advisor = MCAdvisor(config_space)
    assert advisor.config_space == config_space
    assert advisor.num_objectives == 1
    assert advisor.num_constraints == 0
    assert advisor.mc_times == 10
    assert advisor.init_num == 3
    assert advisor.init_strategy == 'random_explore_first'
    assert advisor.optimization_strategy == 'bo'
    assert advisor.surrogate_type == 'gp'
    assert advisor.acq_type == 'mcei'
    assert advisor.acq_optimizer_type == 'batchmc'
    assert advisor.use_trust_region == False
    assert advisor.ref_point == None
    assert advisor.output_dir == 'logs'
    assert advisor.task_id == 'OpenBox'
    assert isinstance(advisor.rng, np.random.RandomState)

    advisor = MCAdvisor(config_space)
    config = advisor.get_suggestion(history=multi_start_history_single_obs)
    assert config is not None

    observation = Observation(config, [0.4], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
    advisor.update_observation(observation)
    assert len(advisor.history) == 1

    configs = advisor.sample_random_configs(config_space, 5)
    assert len(configs) == 5

    history = advisor.get_history()
    assert history == advisor.history

    advisor.save_json("test/datas/test_mc.json")

    advisor.load_json("test/datas/test_mc.json")
