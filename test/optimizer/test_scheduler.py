import pytest

from openbox.optimizer.scheduler import build_scheduler, check_scheduler


def test_fixed_scheduler_stage_params_and_resource_ratio():
    scheduler = build_scheduler(
        'fixed',
        n_resources=[4, 2, 1],
        r_resources=[0.25, 0.5, 1.0],
        fidelity_levels=[0.25, 0.5, 1.0],
    )

    assert scheduler.get_fidelity_levels() == [0.25, 0.5, 1.0]
    assert scheduler.get_stage_params(stage=0) == (4, 0.25)
    assert scheduler.get_stage_params(stage=1) == (2, 0.5)
    assert scheduler.get_stage_params(stage=2) == (1, 1.0)
    assert scheduler.calculate_resource_ratio(0.25) == 0.25
    assert scheduler.calculate_resource_ratio(1.0) == 1.0
    assert scheduler.should_update_history(0.25)
    assert scheduler.should_update_history(1.0)


def test_bohb_scheduler_should_only_update_on_full_fidelity():
    scheduler = build_scheduler('bohb', R=9, eta=3)

    assert scheduler.s_values == [2, 1, 0]
    assert [scheduler.get_bracket_index(i) for i in range(6)] == [2, 1, 0, 2, 1, 0]

    expected_brackets = {
        2: [(9, 1), (3, 3), (1, 9)],
        1: [(5, 3), (1, 9)],
        0: [(3, 9)],
    }
    for s, stages in expected_brackets.items():
        for stage, expected in enumerate(stages):
            n_configs, n_resource = scheduler.get_stage_params(s=s, stage=stage)
            assert (n_configs, n_resource) == expected
            assert scheduler.calculate_resource_ratio(n_resource) == round(n_resource / scheduler.R, 5)

    assert scheduler.calculate_resource_ratio(1) == 0.11111
    assert scheduler.calculate_resource_ratio(3) == 0.33333
    assert scheduler.calculate_resource_ratio(9) == 1.0
    assert scheduler.should_update_history(0.33333) is False
    assert scheduler.should_update_history(1.0) is True


def test_mfes_scheduler_should_update_on_any_fidelity():
    scheduler = build_scheduler('mfes', R=9, eta=3)

    expected_brackets = {
        2: [(9, 1), (3, 3), (1, 9)],
        1: [(5, 3), (1, 9)],
        0: [(3, 9)],
    }
    for s, stages in expected_brackets.items():
        for stage, expected in enumerate(stages):
            assert scheduler.get_stage_params(s=s, stage=stage) == expected

    assert scheduler.should_update_history(0.33333) is True
    assert scheduler.should_update_history(1.0) is True


def test_flatten_scheduler_brackets_and_stage_params():
    scheduler = build_scheduler('flatten', R=9, eta=3, num_nodes=1)

    assert len(scheduler.brackets) == 5
    assert [b['s'] for b in scheduler.brackets] == [2, 1, 0, 0, 0]

    assert scheduler.brackets[0]['stages'] == [(9, 1), (3, 3), (1, 9)]
    assert scheduler.brackets[1]['stages'] == [(5, 3), (1, 9)]
    for bracket in scheduler.brackets[2:]:
        assert bracket['stages'] == [(1, 9)]

    expected_cycle = [2, 1, 0, 0, 0, 2, 1]
    for iter_id, expected_s in enumerate(expected_cycle):
        s = scheduler.get_bracket_index(iter_id)
        assert s == expected_s
        n_configs, n_resource = scheduler.get_stage_params(s=s, stage=0)
        if s == 2:
            assert (n_configs, n_resource) == (9, 1)
        elif s == 1:
            assert (n_configs, n_resource) == (5, 3)
        else:
            assert (n_configs, n_resource) == (1, 9)

    assert scheduler.calculate_resource_ratio(1) == 0.11111
    assert scheduler.calculate_resource_ratio(3) == 0.33333
    assert scheduler.calculate_resource_ratio(9) == 1.0
    assert scheduler.should_update_history(0.33333) is False
    assert scheduler.should_update_history(1.0) is True


def test_mfes_flatten_scheduler_should_update_on_any_fidelity():
    scheduler = build_scheduler('mfes_flatten', R=9, eta=3, num_nodes=1)

    assert len(scheduler.brackets) == 5
    assert [b['s'] for b in scheduler.brackets] == [2, 1, 0, 0, 0]
    assert scheduler.should_update_history(0.33333) is True
    assert scheduler.should_update_history(1.0) is True


def test_check_scheduler_requires_resource_ratio_for_mf():
    def objective_without_resource_ratio(config):
        return [0.0]

    with pytest.raises(ValueError, match='requires objective function to accept "resource_ratio"'):
        check_scheduler(objective_without_resource_ratio, scheduler_type='mfes')

