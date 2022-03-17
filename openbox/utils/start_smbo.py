# License: MIT

from openbox.optimizer import _optimizers
from openbox.utils.config_space.space_utils import get_config_space_from_dict


def create_smbo(objective_func, **kwargs):
    optimizer_name = kwargs['optimizer']
    optimizer_class = _optimizers[optimizer_name]

    config_space = get_config_space_from_dict(kwargs)

    kwargs.pop('optimizer', None)
    kwargs.pop('parameters', None)
    kwargs.pop('conditions', None)

    return optimizer_class(objective_func, config_space, **kwargs)
