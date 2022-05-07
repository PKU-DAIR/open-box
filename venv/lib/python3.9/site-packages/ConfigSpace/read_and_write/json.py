#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from typing import Dict

from ConfigSpace import __version__
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    Hyperparameter,
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
    NormalIntegerHyperparameter,
    NormalFloatHyperparameter,
    OrdinalHyperparameter,
    Constant,
    UnParametrizedHyperparameter,
)
from ConfigSpace.conditions import (
    AbstractCondition,
    EqualsCondition,
    NotEqualsCondition,
    InCondition,
    AndConjunction,
    OrConjunction,
    GreaterThanCondition,
    LessThanCondition,
)
from ConfigSpace.forbidden import (
    ForbiddenEqualsClause,
    ForbiddenAndConjunction,
    ForbiddenInClause,
    AbstractForbiddenComponent,
)


JSON_FORMAT_VERSION = 0.2


################################################################################
# Builder for hyperparameters
def _build_constant(param: Constant) -> Dict:
    return {
        'name': param.name,
        'type': 'constant',
        'value': param.value,
    }


def _build_unparametrized_hyperparameter(param: UnParametrizedHyperparameter) -> Dict:
    return {
        'name': param.name,
        'type': 'unparametrized',
        'value': param.value,
    }


def _build_uniform_float(param: UniformFloatHyperparameter) -> Dict:
    return {
        'name': param.name,
        'type': 'uniform_float',
        'log': param.log,
        'lower': param.lower,
        'upper': param.upper,
        'default': param.default_value
    }


def _build_normal_float(param: NormalFloatHyperparameter) -> Dict:
    return {
        'name': param.name,
        'type': 'normal_float',
        'log': param.log,
        'mu': param.mu,
        'sigma': param.sigma,
        'default': param.default_value
    }


def _build_uniform_int(param: UniformIntegerHyperparameter) -> Dict:
    return {
        'name': param.name,
        'type': 'uniform_int',
        'log': param.log,
        'lower': param.lower,
        'upper': param.upper,
        'default': param.default_value
    }


def _build_normal_int(param: NormalIntegerHyperparameter) -> Dict:
    return {
        'name': param.name,
        'type': 'normal_int',
        'log': param.log,
        'mu': param.mu,
        'sigma': param.sigma,
        'default': param.default_value
    }


def _build_categorical(param: CategoricalHyperparameter) -> Dict:
    return {
        'name': param.name,
        'type': 'categorical',
        'choices': param.choices,
        'default': param.default_value,
        'probabilities': param.probabilities,
    }


def _build_ordinal(param: OrdinalHyperparameter) -> Dict:
    return {
        'name': param.name,
        'type': 'ordinal',
        'sequence': param.sequence,
        'default': param.default_value
    }


################################################################################
# Builder for Conditions
def _build_condition(condition: AbstractCondition) -> Dict:
    if isinstance(condition, AndConjunction):
        return _build_and_conjunction(condition)
    elif isinstance(condition, OrConjunction):
        return _build_or_conjunction(condition)
    elif isinstance(condition, InCondition):
        return _build_in_condition(condition)
    elif isinstance(condition, EqualsCondition):
        return _build_equals_condition(condition)
    elif isinstance(condition, NotEqualsCondition):
        return _build_not_equals_condition(condition)
    elif isinstance(condition, GreaterThanCondition):
        return _build_greater_than_condition(condition)
    elif isinstance(condition, LessThanCondition):
        return _build_less_than_condition(condition)
    else:
        raise TypeError(condition)


def _build_and_conjunction(conjunction: AndConjunction) -> Dict:
    child = conjunction.get_descendant_literal_conditions()[0].child.name
    cond_list = list()
    for component in conjunction.components:
        cond_list.append(_build_condition(component))
    return {
        'child': child,
        'type': 'AND',
        'conditions': cond_list,
    }


def _build_or_conjunction(conjunction: OrConjunction) -> Dict:
    child = conjunction.get_descendant_literal_conditions()[0].child.name
    cond_list = list()
    for component in conjunction.components:
        cond_list.append(_build_condition(component))
    return {
        'child': child,
        'type': 'OR',
        'conditions': cond_list,
    }


def _build_in_condition(condition: InCondition) -> Dict:
    child = condition.child.name
    parent = condition.parent.name
    values = list(condition.values)
    return {
        'child': child,
        'parent': parent,
        'type': 'IN',
        'values': values,
    }


def _build_equals_condition(condition: EqualsCondition) -> Dict:
    child = condition.child.name
    parent = condition.parent.name
    value = condition.value
    return {
        'child': child,
        'parent': parent,
        'type': 'EQ',
        'value': value,
    }


def _build_not_equals_condition(condition: NotEqualsCondition) -> Dict:
    child = condition.child.name
    parent = condition.parent.name
    value = condition.value
    return {
        'child': child,
        'parent': parent,
        'type': 'NEQ',
        'value': value,
    }


def _build_greater_than_condition(condition: GreaterThanCondition) -> Dict:
    child = condition.child.name
    parent = condition.parent.name
    value = condition.value
    return {
        'child': child,
        'parent': parent,
        'type': 'GT',
        'value': value,
    }


def _build_less_than_condition(condition: LessThanCondition) -> Dict:
    child = condition.child.name
    parent = condition.parent.name
    value = condition.value
    return {
        'child': child,
        'parent': parent,
        'type': 'LT',
        'value': value,
    }


################################################################################
# Builder for forbidden
def _build_forbidden(clause) -> Dict:
    if isinstance(clause, ForbiddenEqualsClause):
        return _build_forbidden_equals_clause(clause)
    elif isinstance(clause, ForbiddenInClause):
        return _build_forbidden_in_clause(clause)
    elif isinstance(clause, ForbiddenAndConjunction):
        return _build_forbidden_and_conjunction(clause)
    else:
        raise TypeError(clause)


def _build_forbidden_equals_clause(clause: ForbiddenEqualsClause) -> Dict:
    return {
        'name': clause.hyperparameter.name,
        'type': 'EQUALS',
        'value': clause.value,
    }


def _build_forbidden_in_clause(clause: ForbiddenInClause) -> Dict:
    return {
        'name': clause.hyperparameter.name,
        'type': 'IN',
        # The values are a set, but a set cannot be serialized to json
        'values': list(clause.values),
    }


def _build_forbidden_and_conjunction(clause: ForbiddenAndConjunction) -> Dict:
    return {
        'name': clause.get_descendant_literal_clauses()[0].hyperparameter.name,
        'type': 'AND',
        'clauses': [
            _build_forbidden(component) for component in clause.components
        ],
    }


################################################################################
def write(configuration_space, indent=2):
    """
    Create a string representation of a
    :class:`~ConfigSpace.configuration_space.ConfigurationSpace` in json format.
    This string can be written to file.

    Example
    -------
    .. doctest::

        >>> from ConfigSpace import ConfigurationSpace
        >>> import ConfigSpace.hyperparameters as CSH
        >>> from ConfigSpace.read_and_write import json
        >>> cs = ConfigurationSpace()
        >>> cs.add_hyperparameter(CSH.CategoricalHyperparameter('a', choices=[1, 2, 3]))
        a, Type: Categorical, Choices: {1, 2, 3}, Default: 1

        >>> with open('configspace.json', 'w') as f:
        ...      f.write(json.write(cs))
        305

    Parameters
    ----------
    configuration_space : :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
        a configuration space, which should be written to file.
    indent : int
        number of whitespaces to use as indent

    Returns
    -------
    str
        String representation of the configuration space,
        which will be written to file
    """
    if not isinstance(configuration_space, ConfigurationSpace):
        raise TypeError("pcs_parser.write expects an instance of %s, "
                        "you provided '%s'" % (ConfigurationSpace,
                                               type(configuration_space)))

    hyperparameters = []
    conditions = []
    forbiddens = []

    for hyperparameter in configuration_space.get_hyperparameters():

        if isinstance(hyperparameter, Constant):
            hyperparameters.append(_build_constant(hyperparameter))
        elif isinstance(hyperparameter, UnParametrizedHyperparameter):
            hyperparameters.append(
                _build_unparametrized_hyperparameter(hyperparameter)
            )
        elif isinstance(hyperparameter, UniformFloatHyperparameter):
            hyperparameters.append(_build_uniform_float(hyperparameter))
        elif isinstance(hyperparameter, NormalFloatHyperparameter):
            hyperparameters.append(_build_normal_float(hyperparameter))
        elif isinstance(hyperparameter, UniformIntegerHyperparameter):
            hyperparameters.append(_build_uniform_int(hyperparameter))
        elif isinstance(hyperparameter, NormalIntegerHyperparameter):
            hyperparameters.append(_build_normal_int(hyperparameter))
        elif isinstance(hyperparameter, CategoricalHyperparameter):
            hyperparameters.append(_build_categorical(hyperparameter))
        elif isinstance(hyperparameter, OrdinalHyperparameter):
            hyperparameters.append(_build_ordinal(hyperparameter))
        else:
            raise TypeError(
                "Unknown type: %s (%s)" % (
                    type(hyperparameter), hyperparameter,
                )
            )

    for condition in configuration_space.get_conditions():
        conditions.append(_build_condition(condition))

    for forbidden_clause in configuration_space.get_forbiddens():
        forbiddens.append(_build_forbidden(forbidden_clause))

    rval = {}
    if configuration_space.name is not None:
        rval['name'] = configuration_space.name
    rval['hyperparameters'] = hyperparameters
    rval['conditions'] = conditions
    rval['forbiddens'] = forbiddens
    rval['python_module_version'] = __version__
    rval['json_format_version'] = JSON_FORMAT_VERSION

    return json.dumps(rval, indent=indent)


################################################################################
def read(jason_string):
    """
    Create a configuration space definition from a json string.

    Example
    -------

    .. testsetup:: json_test

        from ConfigSpace import ConfigurationSpace
        import ConfigSpace.hyperparameters as CSH
        from ConfigSpace.read_and_write import json
        cs = ConfigurationSpace()
        cs.add_hyperparameter(CSH.CategoricalHyperparameter('a', choices=[1, 2, 3]))
        with open('configspace.json', 'w') as f:
             f.write(json.write(cs))

    .. doctest:: json_test

        >>> from ConfigSpace.read_and_write import json
        >>> with open('configspace.json', 'r') as f:
        ...     jason_string = f.read()
        ...     config = json.read(jason_string)

    Parameters
    ----------
    jason_string : str
        A json string representing a configuration space definition

    Returns
    -------
    :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
        The deserialized ConfigurationSpace object
    """
    jason = json.loads(jason_string)
    if 'name' in jason:
        configuration_space = ConfigurationSpace(name=jason['name'])
    else:
        configuration_space = ConfigurationSpace()

    for hyperparameter in jason['hyperparameters']:
        configuration_space.add_hyperparameter(_construct_hyperparameter(
            hyperparameter,
        ))

    for condition in jason['conditions']:
        configuration_space.add_condition(_construct_condition(
            condition, configuration_space,
        ))

    for forbidden in jason['forbiddens']:
        configuration_space.add_forbidden_clause(_construct_forbidden(
            forbidden, configuration_space,
        ))

    return configuration_space


def _construct_hyperparameter(hyperparameter: Dict) -> Hyperparameter:
    hp_type = hyperparameter['type']
    name = hyperparameter['name']
    if hp_type == 'constant':
        return Constant(
            name=name,
            value=hyperparameter['value'],
        )
    elif hp_type == 'unparametrized':
        return UnParametrizedHyperparameter(
            name=name,
            value=hyperparameter['value'],
        )
    elif hp_type == 'uniform_float':
        return UniformFloatHyperparameter(
            name=name,
            log=hyperparameter['log'],
            lower=hyperparameter['lower'],
            upper=hyperparameter['upper'],
            default_value=hyperparameter['default'],
        )
    elif hp_type == 'normal_float':
        return NormalFloatHyperparameter(
            name=name,
            log=hyperparameter['log'],
            mu=hyperparameter['mu'],
            sigma=hyperparameter['sigma'],
            default_value=hyperparameter['default'],
        )
    elif hp_type == 'uniform_int':
        return UniformIntegerHyperparameter(
            name=name,
            log=hyperparameter['log'],
            lower=hyperparameter['lower'],
            upper=hyperparameter['upper'],
            default_value=hyperparameter['default'],
        )
    elif hp_type == 'normal_int':
        return NormalIntegerHyperparameter(
            name=name,
            log=hyperparameter['log'],
            lower=hyperparameter['lower'],
            upper=hyperparameter['upper'],
            default_value=hyperparameter['default'],
        )
    elif hp_type == 'categorical':
        return CategoricalHyperparameter(
            name=name,
            choices=hyperparameter['choices'],
            default_value=hyperparameter['default'],
            weights=hyperparameter.get('probabilities'),
        )
    elif hp_type == 'ordinal':
        return OrdinalHyperparameter(
            name=name,
            sequence=hyperparameter['sequence'],
            default_value=hyperparameter['default'],
        )
    else:
        raise ValueError(hp_type)


def _construct_condition(
        condition: Dict,
        cs: ConfigurationSpace,
) -> AbstractCondition:
    condition_type = condition['type']
    if condition_type == 'AND':
        return _construct_and_condition(condition, cs)
    elif condition_type == 'OR':
        return _construct_or_condition(condition, cs)
    elif condition_type == 'IN':
        return _construct_in_condition(condition, cs)
    elif condition_type == 'EQ':
        return _construct_eq_condition(condition, cs)
    elif condition_type == 'NEQ':
        return _construct_neq_condition(condition, cs)
    elif condition_type == 'GT':
        return _construct_gt_condition(condition, cs)
    elif condition_type == 'LT':
        return _construct_lt_condition(condition, cs)
    else:
        raise ValueError(condition_type)


def _construct_and_condition(
        condition: Dict,
        cs: ConfigurationSpace,
) -> AndConjunction:
    conditions = [
        _construct_condition(cond, cs) for cond in condition['conditions']
    ]
    return AndConjunction(*conditions)


def _construct_or_condition(
        condition: Dict,
        cs: ConfigurationSpace,
) -> OrConjunction:
    conditions = [_construct_condition(cond, cs) for cond in condition['conditions']]
    return OrConjunction(*conditions)


def _construct_in_condition(
        condition: Dict,
        cs: ConfigurationSpace,
) -> InCondition:
    return InCondition(
        child=cs.get_hyperparameter(condition['child']),
        parent=cs.get_hyperparameter(condition['parent']),
        values=condition['values'],
    )


def _construct_eq_condition(
        condition: Dict,
        cs: ConfigurationSpace,
) -> EqualsCondition:
    return EqualsCondition(
        child=cs.get_hyperparameter(condition['child']),
        parent=cs.get_hyperparameter(condition['parent']),
        value=condition['value'],
    )


def _construct_neq_condition(
        condition: Dict,
        cs: ConfigurationSpace,
) -> NotEqualsCondition:
    return NotEqualsCondition(
        child=cs.get_hyperparameter(condition['child']),
        parent=cs.get_hyperparameter(condition['parent']),
        value=condition['value'],
    )


def _construct_gt_condition(
        condition: Dict,
        cs: ConfigurationSpace,
) -> GreaterThanCondition:
    return GreaterThanCondition(
        child=cs.get_hyperparameter(condition['child']),
        parent=cs.get_hyperparameter(condition['parent']),
        value=condition['value'],
    )


def _construct_lt_condition(
        condition: Dict,
        cs: ConfigurationSpace,
) -> LessThanCondition:
    return LessThanCondition(
        child=cs.get_hyperparameter(condition['child']),
        parent=cs.get_hyperparameter(condition['parent']),
        value=condition['value'],
    )


def _construct_forbidden(
        clause: Dict,
        cs: ConfigurationSpace,
) -> AbstractForbiddenComponent:
    forbidden_type = clause['type']
    if forbidden_type == 'EQUALS':
        return _construct_forbidden_equals(clause, cs)
    elif forbidden_type == 'IN':
        return _construct_forbidden_in(clause, cs)
    elif forbidden_type == 'AND':
        return _construct_forbidden_and(clause, cs)
    else:
        return ValueError(forbidden_type)


def _construct_forbidden_equals(
        clause: Dict,
        cs: ConfigurationSpace,
) -> ForbiddenEqualsClause:
    return ForbiddenEqualsClause(
        hyperparameter=cs.get_hyperparameter(clause['name']),
        value=clause['value']
    )


def _construct_forbidden_in(
        clause: Dict,
        cs: ConfigurationSpace,
) -> ForbiddenEqualsClause:
    return ForbiddenInClause(
        hyperparameter=cs.get_hyperparameter(clause['name']),
        values=clause['values']
    )


def _construct_forbidden_and(
        clause: Dict,
        cs: ConfigurationSpace,
) -> ForbiddenAndConjunction:
    clauses = [_construct_forbidden(cl, cs) for cl in clause['clauses']]
    return ForbiddenAndConjunction(*clauses)
