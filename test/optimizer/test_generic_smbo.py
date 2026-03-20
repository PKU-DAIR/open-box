import pytest
from openbox.optimizer.base import BOBase
from openbox.optimizer.generic_smbo import SMBO
from openbox.core.generic_advisor import Advisor
from openbox.core.mf_advisor import MFAdvisor
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.history import History, Observation


def _objective_func(config, resource_ratio=1.0):
    return [float(resource_ratio)]


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

        def get_suggestions(self, batch_size=1):
            size = int(batch_size or 1)
            return [self.get_suggestion() for _ in range(size)]

        def update_observation(self, observation, resource_ratio=1.0):
            self.seen_resource_ratios.append(round(float(resource_ratio), 5))
            if resource_ratio == 1.0:
                self.history.update_observation(observation)

    dummy_advisor = DummyAdvisor(configspace_tiny)

    def _build_dummy_advisor(*args, **kwargs):
        return dummy_advisor

    monkeypatch.setattr('openbox.core.build_advisor', _build_dummy_advisor)

    smbo = SMBO(
        objective_function=_objective_func,
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


def test_smbo_batch_sampling(configspace_tiny, monkeypatch):
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
            self.batch_calls = 0
            self.single_calls = 0

        def get_suggestion(self):
            self.single_calls += 1
            return self.config_space.sample_configuration()

        def get_suggestions(self, batch_size=1):
            self.batch_calls += 1
            size = int(batch_size or 1)
            return [self.config_space.sample_configuration() for _ in range(size)]

        def update_observation(self, observation, resource_ratio=1.0):
            if resource_ratio == 1.0:
                self.history.update_observation(observation)

    dummy_advisor = DummyAdvisor(configspace_tiny)

    def _build_dummy_advisor(*args, **kwargs):
        return dummy_advisor

    monkeypatch.setattr('openbox.core.build_advisor', _build_dummy_advisor)

    smbo = SMBO(
        objective_function=_objective_func,
        config_space=configspace_tiny,
        advisor_type='mf',
        scheduler_type='mfes',
        scheduler_kwargs={'R': 9, 'eta': 3},
        max_runs=1,
        initial_runs=0,
        logging_dir='test/datas',
    )
    smbo.run()

    # For first bracket in MFES (s=2), stage0 requests 9 configs once.
    assert dummy_advisor.batch_calls == 1
    assert dummy_advisor.single_calls == 0


def _sample_unique_configs(config_space, n):
    configs = []
    while len(configs) < n:
        conf = config_space.sample_configuration()
        if conf not in configs:
            configs.append(conf)
    return configs


@pytest.mark.parametrize('scheduler_type', ['bohb', 'mfes'])
@pytest.mark.parametrize(
    'R,eta,stage_sizes,stage_ratios',
    [
        (9, 3, [9, 3, 1], [0.11111, 0.33333, 1.0]),
        (27, 3, [27, 9, 3, 1], [0.03704, 0.11111, 0.33333, 1.0]),
    ],
)
def test_smbo_scheduler(
    configspace_tiny, monkeypatch, scheduler_type, R, eta, stage_sizes, stage_ratios
):
    stage0_n = stage_sizes[0]
    stage0_candidates = _sample_unique_configs(configspace_tiny, stage0_n)
    # Make best configs appear at the tail to ensure elimination logic is really exercised.
    score_by_id = {id(cfg): float(stage0_n - i) for i, cfg in enumerate(stage0_candidates)}
    eval_by_ratio = {ratio: [] for ratio in stage_ratios}

    def objective_with_trace(config, resource_ratio=1.0):
        ratio = round(float(resource_ratio), 5)
        eval_by_ratio[ratio].append(config)
        return [score_by_id[id(config)]]

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
            self.batch_calls = []
            self.updated_ratios = []

        def get_suggestion(self):
            return self.config_space.sample_configuration()

        def get_suggestions(self, batch_size=1):
            self.batch_calls.append(int(batch_size))
            if int(batch_size) != stage0_n:
                raise AssertionError(f'Unexpected batch_size: {batch_size}, expected {stage0_n}')
            return list(stage0_candidates)

        def update_observation(self, observation, resource_ratio=1.0):
            ratio = round(float(resource_ratio), 5)
            self.updated_ratios.append(ratio)
            if ratio == 1.0:
                self.history.update_observation(observation)

    dummy_advisor = DummyAdvisor(configspace_tiny)

    def _build_dummy_advisor(*args, **kwargs):
        return dummy_advisor

    monkeypatch.setattr('openbox.core.build_advisor', _build_dummy_advisor)

    smbo = SMBO(
        objective_function=objective_with_trace,
        config_space=configspace_tiny,
        advisor_type='mf',
        scheduler_type=scheduler_type,
        scheduler_kwargs={'R': R, 'eta': eta},
        max_runs=1,
        initial_runs=0,
        logging_dir='test/datas',
    )
    smbo.run()

    # Stage0 should only query advisor once.
    assert dummy_advisor.batch_calls == [stage0_n]

    expected_by_ratio = {}
    current = list(stage0_candidates)
    expected_by_ratio[stage_ratios[0]] = list(current)
    for idx in range(1, len(stage_sizes)):
        keep_n = stage_sizes[idx]
        current = sorted(current, key=lambda c: score_by_id[id(c)])[:keep_n]
        expected_by_ratio[stage_ratios[idx]] = list(current)

    for ratio in stage_ratios:
        assert eval_by_ratio[ratio] == expected_by_ratio[ratio]

    if scheduler_type == 'bohb':
        assert dummy_advisor.updated_ratios == [1.0]
    else:
        expected_updates = []
        for ratio, n in zip(stage_ratios, stage_sizes):
            expected_updates.extend([ratio] * n)
        assert dummy_advisor.updated_ratios == expected_updates


def test_smbo_scheduler_advisor_mapping(configspace_tiny):
    smbo_bohb = SMBO(
        objective_function=_objective_func,
        config_space=configspace_tiny,
        advisor_type='mf',
        scheduler_type='bohb',
        surrogate_type='mfgpe',
        scheduler_kwargs={'R': 9, 'eta': 3},
        max_runs=1,
        initial_runs=0,
        logging_dir='test/datas',
    )
    assert isinstance(smbo_bohb.config_advisor, Advisor)
    assert not isinstance(smbo_bohb.config_advisor, MFAdvisor)
    assert smbo_bohb.config_advisor.surrogate_type != 'mfgpe'

    smbo_mfes = SMBO(
        objective_function=_objective_func,
        config_space=configspace_tiny,
        advisor_type='default',
        scheduler_type='mfes',
        surrogate_type='auto',
        scheduler_kwargs={'R': 9, 'eta': 3},
        max_runs=1,
        initial_runs=0,
        logging_dir='test/datas',
    )
    assert isinstance(smbo_mfes.config_advisor, MFAdvisor)
    assert smbo_mfes.config_advisor.surrogate_type == 'mfgpe'
