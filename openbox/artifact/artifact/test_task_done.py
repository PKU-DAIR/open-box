# License: MIT

import json
import random
import requests
from ConfigSpace.read_and_write import json as config_json
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from openbox.config_space import ConfigurationSpace
from openbox.config_space.util import convert_configurations_to_array

user_id = 18

res = requests.post('http://127.0.0.1:8001/bo_advice/get_suggestion/', data={'id': user_id})
