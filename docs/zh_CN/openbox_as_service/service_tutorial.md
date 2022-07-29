# OpenBox 服务教程

本教程介绍如何使用远程的 **OpenBox** 服务。

## 注册账户

访问 <http://127.0.0.1:11425/user_board/index/> (用你的ip:端口号替换 "127.0.0.1:11425")，你会看到 **OpenBox** 服务的主页。
用Email注册一个账号来使用服务。

你需要点击你邮箱中验证邮件中的链接来激活账号。

## 提交一个任务

这是一个如何用 <font color=#FF0000>**RemoteAdvisor**</font> 和 **OpenBox** 服务交互的实例。

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
    res['objs'] = [-(np.cos((X[0]-0.1)*X[1])**2 + X[0] * np.sin(3*X[0]+X[1]))]
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
    config_advisor.update_observation(config_dict, obs['objs'], obs['constraints'],
                                      trial_info=trial_info, trial_state=SUCCESS)

incumbents, history = config_advisor.get_result()
print(incumbents)
```

+ 在创建**RemoteAdvisor**时，不要忘记设置 **server_ip, port** 和 **email, password**。然后将任务注册到服务器。
  
+ 一旦你创建任务后，你可以通过调用<font color=#FF0000>**RemoteAdvisor.get_suggestion()**</font>来获取配置的建议。

+ 通过调用 <font color=#FF0000>**RemoteAdvisor.update_observation()**</font> 在本地运行作业并将结果发送回服务。 

+ 重复 **get_suggestion** 和 **update_observation** 来完成优化。

如果你对设置问题不熟悉，请参考
[快速上手教程](../quick_start/quick_start).

## 在网页上监控一个任务

您可以随时监视任务并在**OpenBox**服务网页上查看优化结果。

访问 <http://127.0.0.1:11425/user_board/index/> (用你的 ip:port 替换 "127.0.0.1:11425" )
登陆你的账户。

您将找到您创建的所有任务。单击按钮以进一步观察结果并管理任务。

<p align="center">
<img src="https://raw.githubusercontent.com/PKU-DAIR/open-box/master/docs/imgs/user_board_example.png" width="90%">
</p>
