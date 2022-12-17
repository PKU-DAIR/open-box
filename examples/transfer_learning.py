# In this example, we show how to transfer knowledge from source tasks to the target task.

import numpy as np
import matplotlib.pyplot as plt
from openbox import Observation, History, Advisor, space as sp, logger


# Define config space
cs = sp.Space()
for i in range(3):
    cs.add_variable(sp.Float('x%d' % (i+1), -200, 200))


# Define objective function
def obj(config):
    x1, x2, x3 = config['x1'], config['x2'], config['x3']
    y = (x1-10)**2 + x2**2 + (x3-100)**2
    return {'objectives': [y]}


# Generate history data for transfer learning. transfer_learning_history requires a list of History.
transfer_learning_history = list()  # type: List[History]
# 3 source tasks with 50 evaluations of random configurations each
# one task is relevant to the target task, the other two are irrelevant
num_history_tasks, num_results_per_task = 3, 50
for task_idx in range(num_history_tasks):
    # Build a History class for each source task based on previous observations.
    # If source tasks are also optimized by Openbox, you can get the History by
    # using the APIs from Optimizer or Advisor. E.g., history = advisor.get_history()
    history = History(task_id=f'history{task_idx}', config_space=cs)

    for _ in range(num_results_per_task):
        config = cs.sample_configuration()
        if task_idx == 1:  # relevant task
            y = obj(config)['objectives'][0] + 100
        else:              # irrelevant tasks
            y = np.random.random()
        # build and update observation
        observation = Observation(config=config, objectives=[y])
        history.update_observation(observation)

    transfer_learning_history.append(history)


# Define an advisor with an TLBO (Transfer Learning for Bayesian Optimization).
# To switch on TLBO, the input string for surrogate_type includes three parts. The first part must be "tlbo".
# The second part refers to the transfer learning algorithm. So far, we support "rgpe", "sgpr", and "topov3".
# The third part is the surrogate type, e.g., "gp" or "prf". The three parts are joint with "_".
# E.g., "tlbo_rgpe_gp" or "tlbo_sgpr_prf".
# An example of using rgpe as the transfer learning algorithm and gp as the surrogate is shown as follows,

tlbo_advisor = Advisor(
    config_space=cs,
    num_objectives=1,
    num_constraints=0,
    initial_trials=3,
    transfer_learning_history=transfer_learning_history,  # type: List[History]
    surrogate_type='tlbo_rgpe_gp',
    acq_type='ei',
    acq_optimizer_type='random_scipy',
    task_id='TLBO',
)

# Then run the optimization via Advisor APIs. A similar example is provided in ask_and_tell_interface.py
max_iter = 20
for i in range(max_iter):
    config = tlbo_advisor.get_suggestion()
    res = obj(config)
    logger.info(f'Iteration {i+1}, result: {res}')
    observation = Observation(config=config, objectives=res['objectives'])
    tlbo_advisor.update_observation(observation)

# show results
history = tlbo_advisor.get_history()
print(history)
history.plot_convergence()
plt.show()
