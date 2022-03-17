# License: MIT

import os

from openbox.optimizer.base import BOBase
from openbox.utils.class_loader import find_components

"""
Load the buildin optimizers.
"""
optimizers_directory = os.path.split(__file__)[0]
_optimizers = find_components(__package__, optimizers_directory, BOBase)
