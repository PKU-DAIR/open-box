import pytest
from openbox.optimizer.base import BOBase
from openbox.optimizer.generic_smbo import SMBO
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.history import History, Observation


def test_smbo(configspace_tiny, func_brain):
    config_space = configspace_tiny
    objective_function = func_brain
    smbo = SMBO(objective_function, config_space, max_runs=2, initial_runs=1, logging_dir='test/datas')
    assert isinstance(smbo, BOBase)
    assert smbo.objective_function == objective_function
    assert smbo.config_space == config_space

    smbo.run()
    assert smbo.iteration_id == 2
    assert len(smbo.config_advisor.history) == 2


def test_smbo_default(configspace_tiny, func_brain):
    config_space = configspace_tiny
    objective_function = func_brain
    smbo = SMBO(objective_function, config_space, advisor_type='random', max_runs=2, initial_runs=1, logging_dir='test/datas')
    assert isinstance(smbo, BOBase)
    assert smbo.objective_function == objective_function
    assert smbo.config_space == config_space

    smbo.run()
    assert smbo.iteration_id == 2
    assert len(smbo.config_advisor.history) == 2


def test_smbo_multi_fidelity(configspace_tiny, monkeypatch):
    def _obj_func(config, resource_ratio=1.0):
        return [float(resource_ratio)]

    class DummyAdvisor:
        def __init__(self, config_space):
            self.config_space = config_space
            self.init_num = 0
            self.history = History(
                task_id='dummy',
                num_objectives=1,
                num_constraints=0,
                config_space=config_space,
            )
            self.seen_resource_ratios = []

        def get_suggestion(self):
            return self.config_space.sample_configuration()

        def update_observation(self, observation, resource_ratio=1.0):
            self.seen_resource_ratios.append(round(float(resource_ratio), 5))
            if resource_ratio == 1.0:
                self.history.update_observation(observation)

    dummy_advisor = DummyAdvisor(configspace_tiny)

    def _build_dummy_advisor(*args, **kwargs):
        return dummy_advisor

    monkeypatch.setattr('openbox.core.build_advisor', _build_dummy_advisor)

    smbo = SMBO(
        objective_function=_obj_func,
        config_space=configspace_tiny,
        advisor_type='mf',
        scheduler_type='mfes',
        scheduler_kwargs={'R': 9, 'eta': 3},
        max_runs=10,
        initial_runs=0,
        logging_dir='test/datas',
    )
    smbo.run()

    assert smbo.iteration_id == 10
    assert dummy_advisor.seen_resource_ratios.count(0.11111) == 9 * 4
    assert dummy_advisor.seen_resource_ratios.count(0.33333) == 3 * 4 + 5 * 3
    assert dummy_advisor.seen_resource_ratios.count(1.0) == 5 * 3 + 1
    assert 1.0 in dummy_advisor.seen_resource_ratios
    assert any(ratio < 1.0 for ratio in dummy_advisor.seen_resource_ratios)
