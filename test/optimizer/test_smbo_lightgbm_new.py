from openbox.utils.start_smbo import create_smbo
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import balanced_accuracy_score
from lightgbm import LGBMClassifier

# prepare your data
X, y = load_digits(return_X_y=True)
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)


def objective_function(config):
    # convert Configuration to dict
    params = config.get_dictionary()

    # fit model
    model = LGBMClassifier(**params)
    model.fit(x_train, y_train)

    # predict and calculate loss
    y_pred = model.predict(x_val)
    loss = 1 - balanced_accuracy_score(y_val, y_pred)  # OpenBox minimizes the objective

    # return result dictionary
    result = dict(objs=(loss, ))
    return result


config_dict = {
    "optimizer": "SMBO",
    "parameters": {
        "n_estimators": {
            "type": "int",
            "bound": [100, 1000],
            "default": 500,
            "q": 50
        },
        "num_leaves": {
            "type": "int",
            "bound": [31, 2047],
            "default": 128
        },
        "max_depth": {
            "type": "const",
            "value": 15
        },
        "learning_rate": {
            "type": "float",
            "bound": [1e-3, 0.3],
            "default": 0.1,
            "log": True
        },
        "min_child_samples": {
            "type": "int",
            "bound": [5, 30],
            "default": 20
        },
        "subsample": {
            "type": "float",
            "bound": [0.7, 1],
            "default": 1,
            "q": 0.1
        },
        "colsample_bytree": {
            "type": "float",
            "bound": [0.7, 1],
            "default": 1,
            "q": 0.1
        },
    },
    # "num_objs": 1,
    # "num_constraints": 0,
    # "advisor_type": "default",
    "max_runs": 100,
    # "surrogate_type": "prf",
    "time_limit_per_trial": 180,
    # "logging_dir": "logs",
    "task_id": "so_hpo"
}

bo = create_smbo(objective_function, **config_dict)
history = bo.run()

print(history)
history.plot_convergence()
plt.show()

# history.visualize_jupyter()
