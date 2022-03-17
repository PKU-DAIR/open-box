import os
import sys
import bson
import pprint
import datetime
import numpy as np
from bson import ObjectId
sys.path.insert(0, os.getcwd())

from openbox.optimizer import _optimizers
from openbox.artifact.data_manipulation.db_object import User, Task, Runhistory
from openbox.utils.config_space.space_utils import get_config_space_from_dict


def branin(x):
    xs = x.get_dictionary()
    x1 = xs['x1']
    x2 = xs['x2']
    a = 1.
    b = 5.1 / (4. * np.pi ** 2)
    c = 5. / np.pi
    r = 6.
    s = 10.
    t = 1. / (8. * np.pi)
    ret = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    return {'objs': (ret,)}


task_config = {
    "optimizer": "SMBO",
    "parameters": {
        "x1": {
            "type": "float",
            "bound": [-5, 10],
            "default": 0
        },
        "x2": {
            "type": "float",
            "bound": [0, 15]
        },
    },
    "advisor_type": 'default',
    "max_runs": 50,
    "surrogate_type": 'gp',
    "time_limit_per_trial": 5,
    "logging_dir": 'logs',
    "task_id": 'hp1'
}


def test_user():
    user = User()
    for _id in range(5):
        item = {'username': 'username-%d' % _id,
                'email': 'example@pku.edu.cn',
                'password': 'example_pwd',
                'salt': 'example_salt',
                'create_time': datetime.datetime.now()}
        post_id_1 = user.insert_one(item)
        print(post_id_1)
    item = user.find_one({'username': 'username-1'})
    # user.delete_many({'username': 'example_username'})
    pprint.pprint(item)


def test_task():
    task = Task()
    # for _id in range(5):
    #     item = {'task_name': 'task_name-%d' % _id,
    #             'owner': '605c76e8db226d5d47a5b409',
    #             'create_time': datetime.datetime.now(),
    #             'config_space': task_config,
    #             'status': 'running',
    #             'advisor_type': 'smbo',
    #             'max_run': 200,
    #             'surrogate_type': 'gp',
    #             'time_limit_per_trial': 300,
    #             'active_worker_num': 1,
    #             'parallel_type': 'async'
    #             }
    #     post_id_1 = task.insert_one(item)
    #     print(post_id_1)
    item = task.find_one({'task_name': 'task_name-1'})
    pprint.pprint(item)

    items = task.find_all({'owner': ObjectId('605c76e8db226d5d47a5b409')},
                          ['task_name', 'owner', 'create_time', 'config_space',
                           'status', 'advisor_type', 'max_run', 'surrogate_type',
                           'time_limit_per_trial', 'active_worker_num',
                           'parallel_type'])
    pprint.pprint(list(items))

    # Update the task status.
    condition = {"_id": ObjectId('605c7937007a8e5b586e7027')}
    # condition = {"task_name": "task_name-1"}
    task.collection.update_one(condition, {"$set": {"status": "stopped"}})


def test_runhistory():
    runhistory = Runhistory()

    user_id = ObjectId('605c76e8db226d5d47a5b409')
    task_id = ObjectId('605c7937007a8e5b586e7023')
    task_config_ = task_config.copy()
    optimizer_name = task_config_['optimizer']
    optimizer_class = _optimizers[optimizer_name]
    config_space = get_config_space_from_dict(task_config_)
    task_config_.pop('optimizer', None)
    task_config_.pop('parameters', None)
    task_config_.pop('conditions', None)
    optimizer = optimizer_class(branin, config_space, **task_config_)

    # for _id in range(5):
    #     config, trial_state, objs, trial_info = optimizer.iterate()
    #     item = {'user_id': user_id,
    #             'task_id': task_id,
    #             'config': config.get_dictionary(),
    #             'result': objs,
    #             'status': trial_state,
    #             'trial_info': trial_info,
    #             'worker_id': 'worker_1',
    #             'cost': 120}
    #     post_id = runhistory.insert_one(item)
    #     print(post_id)
    item = runhistory.find_all({'user_id': user_id}, ['config', 'result', 'cost'])
    pprint.pprint(list(item))


if __name__ == "__main__":
    # test_user()
    test_task()
    # test_runhistory()
