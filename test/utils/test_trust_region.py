import pytest
from openbox.utils.trust_region import TurboState


def test_turbo_state_initialization_with_valid_input():
    turbo_state = TurboState(5)
    assert turbo_state.dim == 5
    assert turbo_state.failure_tolerance == 5


def test_turbo_state_update_with_improvement():
    turbo_state = TurboState(5)
    turbo_state.best_value = 0.5
    turbo_state.update(0.4)
    assert turbo_state.success_counter == 1
    assert turbo_state.failure_counter == 0
    assert turbo_state.best_value == 0.4


def test_turbo_state_update_without_improvement():
    turbo_state = TurboState(5)
    turbo_state.best_value = 0.5
    turbo_state.update(0.6)
    assert turbo_state.success_counter == 0
    assert turbo_state.failure_counter == 1
    assert turbo_state.best_value == 0.5


def test_turbo_state_update_with_expansion():
    turbo_state = TurboState(5)
    turbo_state.best_value = 0.5
    turbo_state.success_counter = 9
    turbo_state.update(0.4)
    assert turbo_state.length == 1.6
    assert turbo_state.success_counter == 0


def test_turbo_state_update_with_shrinkage():
    turbo_state = TurboState(5)
    turbo_state.best_value = 0.5
    turbo_state.failure_counter = 4
    turbo_state.update(0.6)
    assert turbo_state.length == 0.4
    assert turbo_state.failure_counter == 5


def test_turbo_state_update_with_restart():
    turbo_state = TurboState(5)
    turbo_state.best_value = 0.5
    turbo_state.length = 0.5 ** 7 - 1e-3
    turbo_state.update(0.6)
    assert turbo_state.restart_triggered
