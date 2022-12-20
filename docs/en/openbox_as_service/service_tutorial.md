# OpenBox Service Tutorial

In this tutorial, we will introduce how to use the remote **OpenBox** service.

## Register an Account

Visit <http://127.0.0.1:11425/user_board/index/> (replace "127.0.0.1:11425" with server ip:port) and you will see
the homepage of **OpenBox** service. Register an account by email to use the service.

You need to activate your account by clicking on the link in the activation email.

## Submit a Task

Here is an example of how to use <font color=#FF0000>**RemoteAdvisor**</font> to interact with the **OpenBox** service.

```python
import time
import datetime
import numpy as np


from openbox.artifact.remote_advisor import RemoteAdvisor
from openbox.utils.constants import SUCCESS, FAILED, TIMEOUT, MEMOUT
from openbox.utils.config_space import Configuration, ConfigurationSpace, UniformFloatHyperparameter


# Define objective function to tune
def townsend(config):
    X = np.array(list(config.get_dictionary().values()))
    res = dict()
    res['objectives'] = [-(np.cos((X[0]-0.1)*X[1])**2 + X[0] * np.sin(3*X[0]+X[1]))]
    res['constraints'] = [-(-np.cos(1.5*X[0]+np.pi)*np.cos(1.5*X[1])+np.sin(1.5*X[0]+np.pi)*np.sin(1.5*X[1]))]
    return res


# Define the config space
task_id = time.time()
townsend_params = {
    'float': {
        'x1': (-2.25, 2.5, 0),
        'x2': (-2.5, 1.75, 0)
    }
}
townsend_cs = ConfigurationSpace()
townsend_cs.add_hyperparameters([UniformFloatHyperparameter(e, *townsend_params['float'][e])
                                 for e in townsend_params['float']])


max_runs = 50
# Create remote advisor
config_advisor = RemoteAdvisor(config_space=townsend_cs,
                               server_ip='xx.xx.xx.xx',
                               port=11425,
                               email='xx@xx.com',
                               password='xxxx',
                               num_constraints=1,
                               max_runs=max_runs,
                               acq_type='eic',
                               surrogate_type='gp',
                               task_name="town_send_app")

# Simulate max_runs iterations
for idx in range(max_runs):
    config_dict = config_advisor.get_suggestion()
    config = Configuration(config_advisor.config_space, config_dict)
    print('Get %d config: %s' % (idx+1, config))
    trial_info = {}
    start_time = datetime.datetime.now()
    obs = townsend(config)

    trial_info['cost'] = (datetime.datetime.now() - start_time).seconds
    trial_info['worker_id'] = 0
    trial_info['trial_info'] = 'None'
    print('Result %d is %s. Update observation to server.' % (idx+1, obs))
    config_advisor.update_observation(config_dict, obs['objectives'], obs['constraints'],
                                      trial_info=trial_info, trial_state=SUCCESS)

incumbents, history = config_advisor.get_result()
print(incumbents)
```

+ Remember to set **server_ip, port** of the service and **email, password** of your account when creating 
**RemoteAdvisor**. A task is then registered to the service.

+ Once you create a task, you can get configuration suggestions from the service by calling
<font color=#FF0000>**RemoteAdvisor.get_suggestion()**</font>. 

+ Run your job locally and send results back to the service by calling 
<font color=#FF0000>**RemoteAdvisor.update_observation()**</font>. 

+ Repeat **get_suggestion** and **update_observation** to complete the optimization.

If you are not familiar with setting up a problem, please refer to 
{ref}`Quick Start Tutorial <quick_start/quick_start:quick start>`.

## Monitor a task on the Web Page

You can always monitor your task and watch the optimization results on **OpenBox** service web page.

Visit <http://127.0.0.1:11425/user_board/index/> (replace "127.0.0.1:11425" by server ip:port)
and login your account.

You will find all the tasks you created. Click the buttons to further observe the results and manage your tasks.

<img src="../../imgs/user_board_example.png" width="90%" class="align-center">
