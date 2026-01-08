from openbox.core import build_advisor
from openbox.core.mf_advisor import MFAdvisor
from openbox.utils.constants import SUCCESS
from openbox.utils.history import Observation


def test_mf_advisor_update_observation_by_resource_ratio(configspace_tiny, transfer_learning_history_single):
    advisor = MFAdvisor(
        config_space=configspace_tiny,
        transfer_learning_history=transfer_learning_history_single,
        initial_trials=1,
    )

    config = advisor.get_suggestion()
    low_fidelity_obs = Observation(
        config=config,
        objectives=[0.5],
        trial_state=SUCCESS,
        elapsed_time=0.1,
        extra_info={},
    )
    advisor.update_observation(low_fidelity_obs, resource_ratio=0.5)

    assert len(advisor.history) == 0
    assert len(advisor.history_list) == 1
    assert advisor.resource_identifiers == [0.5]

    full_fidelity_obs = Observation(
        config=config,
        objectives=[0.3],
        trial_state=SUCCESS,
        elapsed_time=0.2,
        extra_info={},
    )
    advisor.update_observation(full_fidelity_obs, resource_ratio=1.0)

    assert len(advisor.history) == 1
    assert len(advisor.history_list) == 2
    assert set(advisor.resource_identifiers) == {0.5, 1.0}


def test_build_mf_advisor_from_factory(configspace_tiny, transfer_learning_history_single):
    advisor = build_advisor(
        advisor_type='mf',
        config_space=configspace_tiny,
        surrogate_type='mfgpe',
        transfer_learning_history=transfer_learning_history_single,
        initial_trials=1,
    )
    assert isinstance(advisor, MFAdvisor)
