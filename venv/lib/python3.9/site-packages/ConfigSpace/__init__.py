# Copyright (c) 2014-2016, ConfigSpace developers
# Matthias Feurer
# Katharina Eggensperger
# and others (see commit history).
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from ConfigSpace.__version__ import __version__
__authors__ = [
    "Matthias Feurer", "Katharina Eggensperger", "Syed Mohsin Ali",
    "Christina Hernandez Wunsch", "Julien-Charles Levesque",
    "Jost Tobias Springenberg", "Philipp Mueller", "Marius Lindauer",
    "Jorn Tuyls"
]

from ConfigSpace.configuration_space import Configuration, \
    ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant, \
    UnParametrizedHyperparameter, OrdinalHyperparameter
from ConfigSpace.conditions import AndConjunction, OrConjunction, \
    EqualsCondition, NotEqualsCondition, InCondition, GreaterThanCondition, LessThanCondition
from ConfigSpace.forbidden import ForbiddenAndConjunction, \
    ForbiddenEqualsClause, ForbiddenInClause

__all__ = ["__version__", "Configuration", "ConfigurationSpace",
           "CategoricalHyperparameter", "UniformFloatHyperparameter",
           "UniformIntegerHyperparameter", "Constant",
           "UnParametrizedHyperparameter", "OrdinalHyperparameter",
           "AndConjunction", "OrConjunction",
           "EqualsCondition", "NotEqualsCondition",
           "InCondition", "GreaterThanCondition",
           "LessThanCondition", "ForbiddenAndConjunction",
           "ForbiddenEqualsClause", "ForbiddenInClause"]
