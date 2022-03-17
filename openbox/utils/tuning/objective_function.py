# License: MIT

from openbox.utils.config_space import Configuration
from sklearn.metrics import balanced_accuracy_score


def get_objective_function(model, x_train, x_val, y_train, y_val, task_type='cls'):
    func_dict = {
        'lightgbm': get_lightgbm_objective_function,
        'xgboost': get_xgboost_objective_function,
    }
    if model not in func_dict.keys():
        raise ValueError('Unsupported model: %s.' % (model,))
    return func_dict[model](x_train, x_val, y_train, y_val, task_type)


def get_lightgbm_objective_function(x_train, x_val, y_train, y_val, task_type='cls'):
    from lightgbm import LGBMClassifier, LGBMRegressor

    def cls_objective_function(config: Configuration):
        # convert Configuration to dict
        params = config.get_dictionary()

        # fit model
        model = LGBMClassifier(**params)
        model.fit(x_train, y_train)

        # predict and calculate loss
        y_pred = model.predict(x_val)
        loss = 1 - balanced_accuracy_score(y_val, y_pred)  # OpenBox minimizes the objective

        # return result dictionary
        result = dict(objs=(loss,))
        return result

    if task_type == 'cls':
        objective_function = cls_objective_function
    elif task_type == 'rgs':
        raise NotImplementedError
    else:
        raise ValueError('Unsupported task type: %s.' % (task_type,))
    return objective_function


def get_xgboost_objective_function(x_train, x_val, y_train, y_val, task_type='cls'):
    from xgboost import XGBClassifier, XGBRegressor

    def cls_objective_function(config: Configuration):
        # convert Configuration to dict
        params = config.get_dictionary()

        # fit model
        model = XGBClassifier(**params, use_label_encoder=False)
        model.fit(x_train, y_train)

        # predict and calculate loss
        y_pred = model.predict(x_val)
        loss = 1 - balanced_accuracy_score(y_val, y_pred)  # OpenBox minimizes the objective

        # return result dictionary
        result = dict(objs=(loss,))
        return result

    if task_type == 'cls':
        objective_function = cls_objective_function
    elif task_type == 'rgs':
        raise NotImplementedError
    else:
        raise ValueError('Unsupported task type: %s.' % (task_type,))
    return objective_function
