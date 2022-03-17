# License: MIT

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant, \
    OrdinalHyperparameter
from ConfigSpace import Configuration
from platypus import Real, Integer
from platypus.operators import CompoundOperator, SBX, PM, HUX, BitFlip
from openbox.utils.util_funcs import get_result


def get_variator(config_space):
    has_int = False
    has_real = False
    for i, param in enumerate(config_space.get_hyperparameters()):
        if isinstance(param, (CategoricalHyperparameter)):
            has_int = True
        elif isinstance(param, (OrdinalHyperparameter)):
            has_int = True
        elif isinstance(param, UniformFloatHyperparameter):
            has_real = True
        elif isinstance(param, UniformIntegerHyperparameter):
            has_real = True
        else:
            raise TypeError("Unsupported hyperparameter type %s" % type(param))
    if has_int and has_real:
        # mixed types
        return CompoundOperator(SBX(), HUX(), PM(), BitFlip())
    else:
        # use default variator
        return None


def set_problem_types(config_space, problem, instance_features=None):
    """
    set problem.types for algorithms in platypus (NSGAII, ...)
    """

    if instance_features is not None:
        raise NotImplementedError
    for i, param in enumerate(config_space.get_hyperparameters()):
        if isinstance(param, (CategoricalHyperparameter)):
            n_cats = len(param.choices)
            problem.types[i] = Integer(0, n_cats - 1)
        elif isinstance(param, (OrdinalHyperparameter)):
            n_cats = len(param.sequence)
            problem.types[i] = Integer(0, n_cats - 1)
        elif isinstance(param, UniformFloatHyperparameter):
            problem.types[i] = Real(0, 1)
        elif isinstance(param, UniformIntegerHyperparameter):
            problem.types[i] = Real(0, 1)
        else:
            raise TypeError("Unsupported hyperparameter type %s" % type(param))


def objective_wrapper(objective_function, config_space, num_constraints):
    def obj_func(x):
        config = Configuration(config_space, vector=x)
        result = objective_function(config)
        objs, constraints = get_result(result)
        if num_constraints > 0:
            return objs, constraints
        else:
            return objs
    return obj_func
