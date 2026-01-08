import pytest

from openbox.optimizer.scheduler import build_scheduler, check_scheduler


def test_fixed_scheduler():
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


def test_bohb_scheduler():
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


def test_mfes_scheduler():
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


def test_flatten_scheduler():
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


def test_mfes_flatten_scheduler():
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


@pytest.mark.parametrize('scheduler_type', ['bohb', 'mfes'])
@pytest.mark.parametrize(
    'R,eta,expected_brackets',
    [
        (
            9,
            3,
            {
                2: [(9, 1), (3, 3), (1, 9)],
                1: [(5, 3), (1, 9)],
                0: [(3, 9)],
            },
        ),
        (
            27,
            3,
            {
                3: [(27, 1), (9, 3), (3, 9), (1, 27)],
                2: [(12, 3), (4, 9), (1, 27)],
                1: [(6, 9), (2, 27)],
                0: [(4, 27)],
            },
        ),
    ],
)
def test_mf_scheduler(
    scheduler_type, R, eta, expected_brackets
):
    scheduler = build_scheduler(scheduler_type, R=R, eta=eta)

    # Bracket scheduling order should be s_max -> ... -> 0 and repeat cyclically.
    expected_cycle = list(reversed(range(scheduler.s_max + 1)))
    assert scheduler.s_values == expected_cycle
    assert [scheduler.get_bracket_index(i) for i in range(len(expected_cycle) * 2)] == expected_cycle * 2

    for s, stages in expected_brackets.items():
        for stage, (expected_n, expected_r) in enumerate(stages):
            n_configs, n_resource = scheduler.get_stage_params(s=s, stage=stage)
            assert (n_configs, n_resource) == (expected_n, expected_r)
            assert scheduler.calculate_resource_ratio(n_resource) == round(expected_r / R, 5)


@pytest.mark.parametrize('scheduler_type', ['bohb', 'mfes'])
@pytest.mark.parametrize(
    'R,eta,bracket_s',
    [
        (9, 3, 2),
        (27, 3, 3),
    ],
)
def test_mf_scheduler_eliminate_candidates(
    scheduler_type, R, eta, bracket_s
):
    scheduler = build_scheduler(scheduler_type, R=R, eta=eta)

    n0, _ = scheduler.get_stage_params(s=bracket_s, stage=0)
    total_n = n0
    candidates = [f'cfg_{i}' for i in range(total_n)]
    perfs = list(reversed(range(n0)))

    current_candidates = candidates
    current_perfs = perfs
    survivors_by_stage = []

    for stage in range(bracket_s):
        next_n, _ = scheduler.get_stage_params(s=bracket_s, stage=stage + 1)
        current_candidates, current_perfs = scheduler.eliminate_candidates(
            current_candidates, current_perfs, s=bracket_s, stage=stage
        )
        survivors_by_stage.append(list(current_candidates))

        assert len(current_candidates) == next_n
        assert len(current_perfs) == next_n
        assert current_perfs == sorted(current_perfs)

    expected_survivors = []
    for stage in range(bracket_s):
        next_n, _ = scheduler.get_stage_params(s=bracket_s, stage=stage + 1)
        expected_survivors.append([f'cfg_{i}' for i in range(total_n - 1, total_n - next_n - 1, -1)])

    assert survivors_by_stage == expected_survivors

