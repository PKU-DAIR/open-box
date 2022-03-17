import numpy as np
from openbox.optimizer.generic_smbo import SMBO
from openbox.utils.config_space import ConfigurationSpace, UniformFloatHyperparameter


def townsend(config):
    config_dict = config.get_dictionary()
    X = np.array([config_dict['x%d' % i] for i in range(2)])
    res = dict()
    res['objs'] = (-(np.cos((X[0]-0.1)*X[1])**2 + X[0] * np.sin(3*X[0]+X[1])), )
    res['constraints'] = (-(-np.cos(1.5*X[0]+np.pi)*np.cos(1.5*X[1])+np.sin(1.5*X[0]+np.pi)*np.sin(1.5*X[1])), )
    return res

townsend_params = {
    'float': {
        'x0': (-2.25, 2.5, 0),
        'x1': (-2.5, 1.75, 0)
    }
}
townsend_cs = ConfigurationSpace()
townsend_cs.add_hyperparameters([UniformFloatHyperparameter(name, *para)
                                 for name, para in townsend_params['float'].items()])


def mishra(config):
    config_dict = config.get_dictionary()
    X = np.array([config_dict['x%d' % i] for i in range(2)])
    x, y = X[0], X[1]
    t1 = np.sin(y) * np.exp((1 - np.cos(x))**2)
    t2 = np.cos(x) * np.exp((1 - np.sin(y))**2)
    t3 = (x - y)**2

    result = dict()
    result['objs'] = [t1 + t2 + t3]
    result['constraints'] = [np.sum((X + 5)**2) - 25]
    return result

mishra_params = {
    'float': {
        'x0': (-10, 0, -5),
        'x1': (-6.5, 0, -3.25)
    }
}
mishra_cs = ConfigurationSpace()
mishra_cs.add_hyperparameters([UniformFloatHyperparameter(name, *para)
                               for name, para in mishra_params['float'].items()])

mishra_optimal_value = -106.7645367

obj_func = mishra
cs = mishra_cs

bo = SMBO(obj_func, cs,
          num_constraints=1,
          num_objs=1,
          acq_optimizer_type='random_scipy',
          max_runs=50,
          task_id='soc')
bo.run()

history = bo.get_history()
print(history)

history.plot_convergence(true_minimum=mishra_optimal_value)
import matplotlib.pyplot as plt
plt.show()
#plt.savefig('logs/plot_convergence_mishra.png')

