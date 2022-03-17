from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, CategoricalHyperparameter, \
    UniformIntegerHyperparameter
from ConfigSpace import EqualsCondition, InCondition, ForbiddenAndConjunction, ForbiddenEqualsClause, ForbiddenInClause

import os
import sys
sys.path.append(os.getcwd())


cs1 = ConfigurationSpace()
x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=0)
x2 = UniformIntegerHyperparameter("x2", 1, 15, q=2.0, log=True)
x3 = CategoricalHyperparameter("x3", ['x1', 'x2', 'x3'])
x4 = CategoricalHyperparameter("x9", ['x1', 'x2', 'x3'])
x5 = CategoricalHyperparameter("x5", ['x1', 'x2', 'x3'])
x6 = CategoricalHyperparameter("x6", ['x1', 'x2', 'x3'])
x7 = CategoricalHyperparameter("x7", ['x1', 'x3', 'x2'], default_value='x2')
cs1.add_hyperparameters([x1, x2, x3, x7, x6, x5, x4])

cond1 = ForbiddenAndConjunction(
    ForbiddenEqualsClause(x3, "x1"),
    ForbiddenEqualsClause(x4, "x2"),
    ForbiddenEqualsClause(x5, "x3")
)
cs1.add_forbidden_clause(cond1)

cond2 = EqualsCondition(x1, x5, "x1")
cond7 = EqualsCondition(x2, x6, "x1")
cond3 = InCondition(x2, x6, ["x1", "x2", "x3"])
cs1.add_condition(cond3)
cs1.add_condition(cond2)

cond4 = ForbiddenEqualsClause(x4, 'x3')
cond5 = ForbiddenInClause(x7, ['x1', 'x3'])
cs1.add_forbidden_clause(cond5)
cs1.add_forbidden_clause(cond4)

from openbox.utils.config_space.space_utils import config_space2string, string2config_space

str = config_space2string(cs1)
print(str)
cs2 = string2config_space(str)
print(cs2 == cs1)
exit()
