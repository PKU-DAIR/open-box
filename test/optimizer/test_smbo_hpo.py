# prepare your data first
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)


from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.config_space import UniformFloatHyperparameter, \
    CategoricalHyperparameter, Constant, UniformIntegerHyperparameter
from sklearn.metrics import balanced_accuracy_score
from lightgbm import LGBMClassifier


def get_configspace():
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


def objective_function(config):
    params = config.get_dictionary()
    params['n_jobs'] = 2
    params['random_state'] = 47

    model = LGBMClassifier(**params)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    loss = 1 - balanced_accuracy_score(y_test, y_pred)  # minimize
    return dict(objs=(loss, ))


from openbox.optimizer.generic_smbo import SMBO
import matplotlib.pyplot as plt

# Run Optimization
bo = SMBO(objective_function,
          get_configspace(),
          num_objs=1,
          num_constraints=0,
          max_runs=100,
          surrogate_type='prf',
          time_limit_per_trial=180,
          task_id='so_hpo')
bo.run()

history = bo.get_history()
print(history)

history.plot_convergence()
#plt.show()
plt.savefig('logs/plot_convergence_hpo.png')

history.visualize_jupyter()
