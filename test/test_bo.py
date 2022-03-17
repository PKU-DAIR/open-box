import os
import sys
sys.path.append(os.getcwd())

import logging
import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score

# Import ConfigSpace and different types of parameters
from openbox.utils.config_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition

from openbox.facade.bo_facade import BayesianOptimization

# We load the iris-dataset (a widely used benchmark)
iris = datasets.load_iris()


def svm_from_cfg(cfg):
    # For deactivated parameters, the configuration stores None-values.
    # This is not accepted by the SVM, so we remove them.
    cfg = {k : cfg[k] for k in cfg if cfg[k]}
    # We translate boolean values:
    cfg["shrinking"] = True if cfg["shrinking"] == "true" else False
    # And for gamma, we set it to a fixed value or to "auto" (if used)
    if "gamma" in cfg:
        cfg["gamma"] = cfg["gamma_value"] if cfg["gamma"] == "value" else "auto"
        cfg.pop("gamma_value", None)  # Remove "gamma_value"

    clf = svm.SVC(**cfg, random_state=42)
    scores = cross_val_score(clf, iris.data, iris.target, cv=5)
    return 1-np.mean(scores)  # Minimize!


logging.basicConfig(level=logging.INFO)

# Build Configuration Space which defines all parameters and their ranges
cs = ConfigurationSpace()

# We define a few possible types of SVM-kernels and add them as "kernel" to our cs
kernel = CategoricalHyperparameter("kernel", ["linear", "rbf", "poly", "sigmoid"], default_value="poly")
cs.add_hyperparameter(kernel)

# There are some hyperparameters shared by all kernels
C = UniformFloatHyperparameter("C", 0.001, 1000.0, default_value=1.0)
shrinking = CategoricalHyperparameter("shrinking", ["true", "false"], default_value="true")
cs.add_hyperparameters([C, shrinking])

# Others are kernel-specific, so we can add conditions to limit the searchspace
degree = UniformIntegerHyperparameter("degree", 1, 5, default_value=3)     # Only used by kernel poly
coef0 = UniformFloatHyperparameter("coef0", 0.0, 10.0, default_value=0.0)  # poly, sigmoid
cs.add_hyperparameters([degree, coef0])
use_degree = InCondition(child=degree, parent=kernel, values=["poly"])
use_coef0 = InCondition(child=coef0, parent=kernel, values=["poly", "sigmoid"])
cs.add_conditions([use_degree, use_coef0])

# This also works for parameters that are a mix of categorical and values from a range of numbers
# For example, gamma can be either "auto" or a fixed float
gamma = CategoricalHyperparameter("gamma", ["auto", "value"], default_value="auto")  # only rbf, poly, sigmoid
gamma_value = UniformFloatHyperparameter("gamma_value", 0.0001, 8, default_value=1)
cs.add_hyperparameters([gamma, gamma_value])
# We only activate gamma_value if gamma is set to "value"
cs.add_condition(InCondition(child=gamma_value, parent=gamma, values=["value"]))
# And again we can restrict the use of gamma in general to the choice of the kernel
cs.add_condition(InCondition(child=gamma, parent=kernel, values=["rbf", "poly", "sigmoid"]))

# Example call of the function
# It returns: Status, Cost, Runtime, Additional Infos
def_value = svm_from_cfg(cs.get_default_configuration())
print("Default Value: %.2f" % (def_value))

# Optimize, using a SMAC-object
print("Optimizing! Depending on your machine, this might take a few minutes.")
bo = BayesianOptimization(svm_from_cfg, cs, max_runs=30, time_limit_per_trial=30, logging_dir='logs')
bo.run()
inc_value = bo.get_incumbent()

print(inc_value)
