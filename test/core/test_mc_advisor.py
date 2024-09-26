import pytest
import numpy as np
from openbox.core.mc_advisor import MCAdvisor
from openbox.utils.early_stop import EarlyStopException
from openbox.utils.history import History, Observation
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



def test_mc_advisor_early_stop(configspace_tiny):
    config_space = configspace_tiny
    advisor = MCAdvisor(config_space, early_stop=True, early_stop_kwargs={'min_iter': 3, 'max_no_improvement_rounds': 1})

    for i in range(3):
        suggestion = advisor.get_suggestion()
        observation = Observation(suggestion, [1], trial_state=SUCCESS, elapsed_time=2.0, extra_info={})
        advisor.update_observation(observation)

    with pytest.raises(EarlyStopException):
        advisor.get_suggestion()

def test_mc_different_constraints_and_objectives(configspace_tiny, multi_start_history_double_obs):
    config_space = configspace_tiny

    Ob_array = [1, 2, 2, 5, 5]
    Co_array = [1, 0, 1, 0, 1]
    Ty_array = ['mceic', 'mcehvi', 'mcehvic', 'mcparego', 'mcparegoc']

    for (o,c,t) in zip(Ob_array, Co_array, Ty_array):
        ref_point = None
        if 'ehvi' in t:
            ref_point = [0.1,0.2]
            ad = MCAdvisor(config_space, o, c, ref_point=ref_point)
            if c == 0:
                ad.get_suggestion(multi_start_history_double_obs)
        else:
            ad = MCAdvisor(config_space, o, c, ref_point=ref_point)
        assert ad.acq_type == t