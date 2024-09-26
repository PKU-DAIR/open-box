import pytest
from openbox.utils.early_stop import EarlyStopAlgorithm
from openbox.core.generic_advisor import Advisor
from openbox import get_config_space
from typing import List

class HistoryMock:
    def __init__(self, observations=None):
        self.observations = observations
        self.meta_info = dict()

    def __len__(self):
        return len(self.observations)
    
    @property
    def objectives(self) -> List[List[float]]:
        return self.observations

    def get_incumbent_value(self):
        return min([obs[0] for obs in self.observations])

def test_es_init():
    # Test initialization
    early_stop = EarlyStopAlgorithm(min_iter=5, min_improvement_percentage=0.1, max_no_improvement_rounds=3)
    assert early_stop.min_iter == 5
    assert early_stop.min_improvement_percentage == 0.1
    assert early_stop.max_no_improvement_rounds == 3

def test_es_check_setup():
    with pytest.raises(ValueError):
        advisor = Advisor(config_space=get_config_space('lightgbm'), num_objectives = 2, early_stop=True, acq_type="parego")
    
    advisor = Advisor(config_space=get_config_space('lightgbm'), num_objectives = 1, early_stop=True, acq_type="ei")
    with pytest.raises(AssertionError):
        early_stop = EarlyStopAlgorithm(min_iter=0)
        early_stop.check_setup(advisor)
    
    with pytest.raises(AssertionError):
        early_stop = EarlyStopAlgorithm(min_improvement_percentage=-0.1)
        early_stop.check_setup(advisor)
    
    with pytest.raises(AssertionError):
        early_stop = EarlyStopAlgorithm(max_no_improvement_rounds=-3)
        early_stop.check_setup(advisor)
    
    with pytest.raises(AssertionError):
        early_stop = EarlyStopAlgorithm()
        advisor.acq_type="eips"
        early_stop.check_setup(advisor)

def test_es_already_early_stopped():
    history = HistoryMock()
    early_stop = EarlyStopAlgorithm()
    assert not early_stop.already_early_stopped(history)

    early_stop.set_already_early_stopped(history)
    assert early_stop.already_early_stopped(history)

    assert early_stop.decide_early_stop_before_suggest(history)

def test_es_decide_early_stop_before_suggest():
    # check not reach min
    history = HistoryMock()
    early_stop = EarlyStopAlgorithm(min_iter=5)
    history.observations = [[1], [1], [1]]
    assert not early_stop.decide_early_stop_before_suggest(history)

    # check not enabled
    early_stop.min_iter=1
    early_stop.max_no_improvement_rounds = 0
    history.observations = [[1], [1], [1]]
    assert not early_stop.decide_early_stop_before_suggest(history)

    # enabled
    early_stop.max_no_improvement_rounds = 2
    history.observations = [[5], [4], [3], [3], [3], [2]]
    assert not early_stop.decide_early_stop_before_suggest(history)
    history.observations = [[5], [4], [3], [3], [3], [3]]
    assert early_stop.decide_early_stop_before_suggest(history)

def test_es_decide_early_stop_after_suggest():
    # check not reach min
    history = HistoryMock()
    early_stop = EarlyStopAlgorithm(min_iter=5, max_no_improvement_rounds=0)
    history.observations = [[1], [1], [1]]
    assert not early_stop.decide_early_stop_after_suggest(history)

    # check not enabled
    early_stop.min_iter=1
    early_stop.min_improvement_percentage=0
    history.observations = [[1], [1], [1]]
    assert not early_stop.decide_early_stop_after_suggest(history, max_acq_value=0)

    # enabled
    history = HistoryMock()
    early_stop.min_improvement_percentage=0.5
    history.observations = [[10], [9], [8]]
    assert early_stop.decide_early_stop_after_suggest(history, max_acq_value=0)