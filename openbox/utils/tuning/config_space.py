# License: MIT

from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.config_space import UniformFloatHyperparameter, \
    CategoricalHyperparameter, Constant, UniformIntegerHyperparameter


def get_config_space(model, task_type='cls'):
    func_dict = {
        'lightgbm': get_lightgbm_config_space,
        'xgboost': get_xgboost_config_space,
    }
    if model not in func_dict.keys():
        raise ValueError('Unsupported model: %s.' % (model, ))
    return func_dict[model](task_type)


def get_lightgbm_config_space(task_type='cls'):
    if task_type == 'cls':
        cs = ConfigurationSpace()
        n_estimators = UniformIntegerHyperparameter("n_estimators", 100, 1000, default_value=500, q=50)
        num_leaves = UniformIntegerHyperparameter("num_leaves", 31, 2047, default_value=128)
        max_depth = Constant('max_depth', 15)
        learning_rate = UniformFloatHyperparameter("learning_rate", 1e-3, 0.3, default_value=0.1, log=True)
        min_child_samples = UniformIntegerHyperparameter("min_child_samples", 5, 30, default_value=20)
        subsample = UniformFloatHyperparameter("subsample", 0.7, 1, default_value=1, q=0.1)
        colsample_bytree = UniformFloatHyperparameter("colsample_bytree", 0.7, 1, default_value=1, q=0.1)
        cs.add_hyperparameters([n_estimators, num_leaves, max_depth, learning_rate, min_child_samples, subsample,
                                colsample_bytree])
        return cs
    elif task_type == 'rgs':
        raise NotImplementedError
    else:
        raise ValueError('Unsupported task type: %s.' % (task_type, ))


def get_xgboost_config_space(task_type='cls'):
    if task_type == 'cls':
        cs = ConfigurationSpace()
        n_estimators = UniformIntegerHyperparameter("n_estimators", 100, 1000, q=50, default_value=500)
        max_depth = UniformIntegerHyperparameter("max_depth", 1, 12)
        learning_rate = UniformFloatHyperparameter("learning_rate", 1e-3, 0.9, log=True, default_value=0.1)
        min_child_weight = UniformFloatHyperparameter("min_child_weight", 0, 10, q=0.1, default_value=1)
        subsample = UniformFloatHyperparameter("subsample", 0.1, 1, q=0.1, default_value=1)
        colsample_bytree = UniformFloatHyperparameter("colsample_bytree", 0.1, 1, q=0.1, default_value=1)
        gamma = UniformFloatHyperparameter("gamma", 0, 10, q=0.1, default_value=0)
        reg_alpha = UniformFloatHyperparameter("reg_alpha", 0, 10, q=0.1, default_value=0)
        reg_lambda = UniformFloatHyperparameter("reg_lambda", 1, 10, q=0.1, default_value=1)
        cs.add_hyperparameters([n_estimators, max_depth, learning_rate, min_child_weight, subsample,
                                colsample_bytree, gamma, reg_alpha, reg_lambda])
        return cs
    elif task_type == 'rgs':
        raise NotImplementedError
    else:
        raise ValueError('Unsupported task type: %s.' % (task_type, ))
