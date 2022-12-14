# In this example, we show how to transfer knowledge from two source tasks to the target task (branin_target).

from openbox.core.generic_advisor import Advisor
from openbox import Observation, History

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
import numpy as np

cs = ConfigurationSpace()
hp1 = UniformFloatHyperparameter('x1', -5, 10)
hp2 = UniformFloatHyperparameter('x2', 0, 15)
cs.add_hyperparameters([hp1, hp2])


# The source objective functions, which are slightly different from the target function.
def branin_source_1(config):
    x1, x2 = config['x1'], config['x2']
    y = (2 * x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2 \
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    return y


def branin_source_2(config):
    x1, x2 = config['x1'], config['x2']
    y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2 \
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10 + 10
    return y

# The target objective function.
def branin_target(config):
    x1, x2 = config['x1'], config['x2']
    y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2 \
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    return y


# In both source tasks, we assume we have already evaluated 50 random configurations.
num_source_samples = 50
source_configs_1 = cs.sample_configuration(num_source_samples)
source_perfs_1 = [branin_source_1(config) for config in source_configs_1]

# Build a History class for each source task based on previous observations.
# If source tasks are also optimized by Openbox, you can get the History by using the APIs from Optimizer or Advisor.
history_1 = History(task_id='history1', config_space=cs)
for i, _ in enumerate(source_configs_1):
    observation = Observation(config=source_configs_1[i], objectives=[source_perfs_1[i]])
    history_1.update_observation(observation)

source_configs_2 = cs.sample_configuration(num_source_samples)
source_perfs_2 = [branin_source_2(config) for config in source_configs_2]

history_2 = History(task_id='history2', config_space=cs)
for i, _ in enumerate(source_configs_2):
    observation = Observation(config=source_configs_2[i], objectives=[source_perfs_2[i]])
    history_2.update_observation(observation)

# Define an advisor with an TLBO (Transfer Learning for Bayesian Optimization).
# history_bo_data requires a list of History,
# To switch on TLBO, the input string for surrogate_type includes three parts. The first part must be "tlbo".
# The second part refers to the transfer learning algorithm. So far, we support "rgpe", "sgpr", and "topov3".
# The third part is the surrogate type, e.g., "gp" or "prf". The three parts are joint with "_".
# An example of using rgpe as the transfer learning algorithm and gp as the surrogate is shown as follows,

rgpe_advisor = Advisor(cs, num_objectives=1, num_constraints=0, initial_trials=5,
                       history_bo_data=[history_1, history_2],
                       surrogate_type='tlbo_rgpe_gp', acq_type='ei', acq_optimizer_type='random_scipy')

# Then run the optimization via Advisor APIs. A similar example is provided in ask_and_tell_interface.py
iteration_budget = 50
for i in range(iteration_budget):
    config = rgpe_advisor.get_suggestion()
    result = branin_target(config)
    print('The %d-th iteration: Result %f' % (i, result))
    observation = Observation(config=config, objectives=[result])
    rgpe_advisor.update_observation(observation)
