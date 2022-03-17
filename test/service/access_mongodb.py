import os
import pprint
import pymongo
import datetime
import numpy as np
import configparser
from openbox.optimizer import _optimizers
from openbox.utils.start_smbo import create_smbo
from openbox.utils.config_space.space_utils import get_config_space_from_dict

# Read configuration from file.
conf_dir = './conf'
config_path = os.path.join(conf_dir, 'service.conf')
config = configparser.ConfigParser()
config.read(config_path)
name_server = dict(config.items('database'))
host = name_server['database_address']
port = name_server['database_port']
username = name_server['user']
password = name_server['password']
my_url = 'mongodb://' + username + ':' + password + '@%s:%s/' % (host, port)

# Connect to the local MongoDB
myclient = pymongo.MongoClient(my_url)

mydb = myclient[username]


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


def test_insertion():
    # Create table & insert data
    user_collection = mydb.user_collection # creating a new table (so easy)

    post1 = {'id': 0, 'username': 'example_username',
             'email': 'example_email@pku.edu.cn', 'pwd': 'example_pwd',
             'salt': 'example_salt'}

    post_id_1 = user_collection.insert_one(post1).inserted_id
    item = user_collection.find_one({'username':'example_username'})
    pprint.pprint(item)


def test_task_manipulation():
    """
        MongoDB command: db.tasks.find()
    Returns
    -------

    """

    # Create table & insert data
    task_collection = mydb.tasks

    new_task = {'task_name': 'quick_start', 'task_config': task_config}

    _ = task_collection.insert_one(new_task).inserted_id
    item = task_collection.find_one({'task_name': 'quick_start'})
    pprint.pprint(item)
    print(type(item))


def test_task_manipulation1():
    """
        Show Usage about Runhistory.
    Returns
    -------

    """
    runhistory_collection = mydb.runhistory
    optimizer_name = task_config['optimizer']
    optimizer_class = _optimizers[optimizer_name]
    config_space = get_config_space_from_dict(task_config)
    task_config.pop('optimizer', None)
    task_config.pop('parameters', None)
    task_config.pop('conditions', None)
    optimizer = optimizer_class(branin, config_space, **task_config)

    for _ in range(10):
        config, trial_state, objs, trial_info = optimizer.iterate()
        print(config, objs)
        new_history = {'task_id': 'abc', 'config': config.get_dictionary(), 'result': list(objs), 'trial_status': trial_state}
        id_ = runhistory_collection.insert_one(new_history).inserted_id
        print(id_)
    item = runhistory_collection.find_one({'task_id': 'abc'})
    pprint.pprint(item)


if __name__ == "__main__":
    # test_insertion()
    # test_task_manipulation()
    test_task_manipulation1()
