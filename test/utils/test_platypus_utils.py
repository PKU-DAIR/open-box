import pytest
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant, \
    OrdinalHyperparameter
from ConfigSpace import Configuration, ConfigurationSpace
from platypus import Problem, Real, Integer
from openbox.utils.platypus_utils import get_variator, set_problem_types, objective_wrapper


def test_variator_with_mixed_hyperparameters():
    config_space = ConfigurationSpace()
    config_space.add_hyperparameter(CategoricalHyperparameter("param1", ["option1", "option2"]))
    config_space.add_hyperparameter(UniformFloatHyperparameter("param2", 0.0, 1.0))
    variator = get_variator(config_space)
    assert variator is not None


def test_variator_with_single_hyperparameter_type():
    config_space = ConfigurationSpace()
    config_space.add_hyperparameter(CategoricalHyperparameter("param1", ["option1", "option2"]))
    variator = get_variator(config_space)
    assert variator is None


def test_problem_types_with_various_hyperparameters():
    config_space = ConfigurationSpace()
    config_space.add_hyperparameter(CategoricalHyperparameter("param1", ["option1", "option2"]))
    config_space.add_hyperparameter(UniformFloatHyperparameter("param2", 0.0, 1.0))
    problem = Problem(2, 1)
    set_problem_types(config_space, problem)
    assert len(problem.types) == 2


def test_objective_wrapper_with_constraints():
    config_space = ConfigurationSpace()
    config_space.add_hyperparameter(CategoricalHyperparameter("param1", ["option1", "option2"]))

    def objective_function(config):
        return {'objectives': [1.0], 'constraints': [0.0], 'extra_info': {}}

    obj_func = objective_wrapper(objective_function, config_space, 1)
    objectives, constraints = obj_func([0])
    assert objectives == [1.0]
    assert constraints == [0.0]


def test_objective_wrapper_without_constraints():
    config_space = ConfigurationSpace()
    config_space.add_hyperparameter(CategoricalHyperparameter("param1", ["option1", "option2"]))

    def objective_function(config):
        return {'objectives': [1.0], 'constraints': [], 'extra_info': {}}

    obj_func = objective_wrapper(objective_function, config_space, 0)
    objectives = obj_func([0])
    assert objectives == [1.0]
