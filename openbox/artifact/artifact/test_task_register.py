# License: MIT

import json
import random
import requests
from ConfigSpace.read_and_write import json as config_json
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from openbox.config_space import ConfigurationSpace
from openbox.config_space.util import convert_configurations_to_array

user_id = 18

cs = ConfigurationSpace()
x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=0)
x2 = UniformFloatHyperparameter("x2", 0, 15, default_value=0)
cs.add_hyperparameters([x1, x2])

config_space_array = config_json.write(cs)

res = requests.post('http://127.0.0.1:8001/bo_advice/task_register/',
                    data={'id':user_id, 'config_space_array':config_space_array})
print('-----------------')
print(res)
print('-----------------')
print(res.text)
print('-----------------')
