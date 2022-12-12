# License: MIT

import typing
import numbers
import functools
import numpy as np
import numpy.random.mtrand

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant, \
    OrdinalHyperparameter
from openbox import logger
from openbox.utils.constants import MAXINT


def get_types(config_space, instance_features=None):
    """TODO"""
    # Extract types vector for rf from config space and the bounds
    types = np.zeros(len(config_space.get_hyperparameters()),
                     dtype=np.uint)
    bounds = [(np.nan, np.nan)]*types.shape[0]

    for i, param in enumerate(config_space.get_hyperparameters()):
        if isinstance(param, (CategoricalHyperparameter)):
            n_cats = len(param.choices)
            types[i] = n_cats
            bounds[i] = (int(n_cats), np.nan)

        elif isinstance(param, (OrdinalHyperparameter)):
            n_cats = len(param.sequence)
            types[i] = 0
            bounds[i] = (0, int(n_cats) - 1)

        elif isinstance(param, Constant):
            # for constants we simply set types to 0
            # which makes it a numerical parameter
            types[i] = 0
            bounds[i] = (0, np.nan)
            # and we leave the bounds to be 0 for now
        elif isinstance(param, UniformFloatHyperparameter):         # Are sampled on the unit hypercube thus the bounds
            # bounds[i] = (float(param.lower), float(param.upper))  # are always 0.0, 1.0
            bounds[i] = (0.0, 1.0)
        elif isinstance(param, UniformIntegerHyperparameter):
            # bounds[i] = (int(param.lower), int(param.upper))
            bounds[i] = (0.0, 1.0)
        elif not isinstance(param, (UniformFloatHyperparameter,
                                    UniformIntegerHyperparameter,
                                    OrdinalHyperparameter)):
            raise TypeError("Unknown hyperparameter type %s" % type(param))

    if instance_features is not None:
        types = np.hstack(
            (types, np.zeros((instance_features.shape[1]))))

    types = np.array(types, dtype=np.uint)
    bounds = np.array(bounds, dtype=object)
    return types, bounds


def transform_to_1d_list(x, hint='result', dtype=np.float64):
    """
    Transform a scalar, 1-d list, tuple, np.ndarray, touch.Tensor to 1-d list
    If x is None or x is not 1-d (after squeeze), raise an error
    """
    assert x is not None, f'{hint} is None!'
    x = np.asarray(x, dtype=dtype)
    original_shape = x.shape
    x = np.squeeze(x)  # np.squeeze requires numpy>=1.7.0
    if x.ndim == 0:
        x = x.reshape(1)
    assert x.ndim == 1, f'The {hint} should be a 1-D array, but got shape: {x.shape}'
    if x.shape != original_shape:
        logger.warning(f'The shape of {hint} is changed from {original_shape} to {x.shape}.')
    return x.tolist()


def parse_result(result):
    """
    Parse (objectives, constraints, extra_info) from result returned by objective function.

    Parameters
    ----------
    result: dict (or float, list, np.ndarray)
        The result returned by objective function.
        Dict is recommended, but we try to support other types, such as float, list, np.ndarray, etc.
        If result is a dict, it should contain at least one key "objectives" (or "objs" for backward compatibility).
        Optional keys: "constraints", "extra_info".

    Returns
    -------
    objectives: list
        The list of objectives.
    constraints: list, optional
        The list of constraints.
    extra_info: dict, optional
        The extra information.
    """
    constraints, extra_info = None, None
    if result is None:
        raise ValueError('result is None!')
    elif isinstance(result, dict):  # recommended usage
        # for backward compatibility
        objectives = result.pop('objectives', None)
        objs = result.pop('objs', None)  # todo: deprecated
        if objectives is not None and objs is not None:
            raise ValueError('"objectives" and "objs" are both provided! Please only provide "objectives".')
        elif objectives is None and objs is None:
            raise ValueError('"objectives" is None!')
        elif objectives is None:
            objectives = objs
            logger.warning('Provide "objs" in result is deprecated and will be removed in future versions! '
                           'Please use "objectives" instead.')

        # objectives is not None now
        objectives = transform_to_1d_list(objectives, hint='objectives')

        # optional keys
        constraints = result.pop('constraints', None)
        if constraints is not None:
            constraints = transform_to_1d_list(constraints, hint='constraints')
        extra_info = result.pop('extra_info', None)
        if len(result) > 0:
            logger.warning(f'Unused information in result: {result}')
    else:
        # provide only objectives
        logger.warning(f'Provide result as <dict> that contains "objectives" is recommended, got {type(result)}')
        objectives = transform_to_1d_list(result, hint='objectives')

    return objectives, constraints, extra_info


def check_random_state(seed):
    """ from [sklearn.utils.check_random_state]
    Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def get_rng(
        rng: typing.Optional[typing.Union[int, np.random.RandomState]] = None,
        run_id: typing.Optional[int] = None,
) -> typing.Tuple[int, np.random.RandomState]:
    """
    Initialize random number generator and set run_id

    * If rng and run_id are None, initialize a new generator and sample a run_id
    * If rng is None and a run_id is given, use the run_id to initialize the rng
    * If rng is an int, a RandomState object is created from that.
    * If rng is RandomState, return it
    * If only run_id is None, a run_id is sampled from the random state.

    Parameters
    ----------
    rng : np.random.RandomState|int|None
    run_id : int, optional

    Returns
    -------
    int
    np.random.RandomState

    """
    # initialize random number generator
    if rng is not None and not isinstance(rng, (int, np.random.RandomState)):
        raise TypeError('Argument rng accepts only arguments of type None, int or np.random.RandomState, '
                        'you provided %s.' % str(type(rng)))
    if run_id is not None and not isinstance(run_id, int):
        raise TypeError('Argument run_id accepts only arguments of type None, int or np.random.RandomState, '
                        'you provided %s.' % str(type(run_id)))

    if rng is None and run_id is None:
        # Case that both are None
        logger.debug('No rng and no run_id given: using a random value to initialize run_id.')
        rng = np.random.RandomState()
        run_id = rng.randint(MAXINT)
    elif rng is None and isinstance(run_id, int):
        logger.debug('No rng and no run_id given: using run_id %d as seed.', run_id)
        rng = np.random.RandomState(seed=run_id)
    elif isinstance(rng, int):
        if run_id is None:
            run_id = rng
        else:
            pass
        rng = np.random.RandomState(seed=rng)
    elif isinstance(rng, np.random.RandomState):
        if run_id is None:
            run_id = rng.randint(MAXINT)
        else:
            pass
    else:
        raise ValueError('This should not happen! Please contact the developers! Arguments: rng=%s of type %s and '
                         'run_id=% of type %s' % (rng, type(rng), run_id, type(run_id)))
    return run_id, rng


def deprecate_kwarg(old_name, new_name, removed_version='a future version'):
    """
    Returns a decorator to deprecate a keyword argument in a function.
    """
    assert old_name != new_name

    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            if old_name in kwargs and new_name in kwargs:
                raise TypeError(
                    'Keyword argument "%s" is deprecated and will be removed in %s. '
                    'Cannot use both kwargs "%s" and "%s"!' % (old_name, removed_version, old_name, new_name))

            if old_name in kwargs:
                logger.warning('Keyword argument "%s" is deprecated and will be removed in %s. '
                               'Please use "%s" instead.' % (old_name, removed_version, new_name))
                kwargs[new_name] = kwargs.pop(old_name)
            return func(*args, **kwargs)
        return wrapped_func
    return decorator
