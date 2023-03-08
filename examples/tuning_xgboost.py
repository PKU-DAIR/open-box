# License: MIT

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

from openbox import get_config_space, get_objective_function
from openbox import Optimizer


if __name__ == "__main__":
    # prepare your data
    X, y = load_digits(return_X_y=True)
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

    # get config_space and objective_function
    config_space = get_config_space('xgboost')
    objective_function = get_objective_function('xgboost', x_train, x_val, y_train, y_val)

    # run
    opt = Optimizer(
        objective_function,
        config_space,
        max_runs=100,
        surrogate_type='prf',
        task_id='tuning_xgboost',
        # Have a try on the new HTML visualization feature!
        # visualization='advanced',   # or 'basic'. For 'advanced', run 'pip install "openbox[extra]"' first
        # auto_open_html=True,        # open the visualization page in your browser automatically
    )
    history = opt.run()

    print(history)

    history.plot_convergence()
    plt.show()

    # install pyrfr to use get_importance()
    print(history.get_importance())

    # Have a try on the new HTML visualization feature!
    # You can also call visualize_html() after optimization.
    # For 'show_importance' and 'verify_surrogate', run 'pip install "openbox[extra]"' first
    # history.visualize_html(open_html=True, show_importance=True, verify_surrogate=True, optimizer=opt)
