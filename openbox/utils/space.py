# License: MIT

# More readable wrapped classes based on ConfigSpace.
# The comments are modified based on https://github.com/automl/ConfigSpace/blob/master/ConfigSpace/hyperparameters.pyx
from typing import List, Dict, Tuple, Union, Optional, Callable
import numpy as np
import ConfigSpace as CS
from ConfigSpace import EqualsCondition, InCondition, ForbiddenAndConjunction, ForbiddenEqualsClause, ForbiddenInClause
from ConfigSpace import ConfigurationSpace, Configuration
from ConfigSpace.exceptions import ForbiddenValueError


# Hyperparameters
class Variable:
    pass


class Int(CS.UniformIntegerHyperparameter, Variable):
    def __init__(self, name: str, lower: int, upper: int, default_value: Union[int, None] = None,
                 q: Union[int, None] = None, log: bool = False, meta: Optional[Dict] = None) -> None:
        """
        An integer variable.

        Its values are sampled from a uniform distribution
        with bounds ``lower`` and ``upper``.

        Example
        -------

        >>> import openbox.utils.space as sp
        >>> cs = sp.Space(seed=1)
        >>> uniform_integer = sp.Int(name='uni_int', lower=10, upper=100, log=False)
        >>> cs.add_variable(uniform_integer)
        uni_int, Type: Int, Range: [10, 100], Default: 55

        Parameters
        ----------
        name : str
            Name of the variable with which it can be accessed
        lower : int
            Lower bound of a range of values from which the variable will be sampled
        upper : int
            upper bound
        default_value : int, optional
            Sets the default value of a variable to a given value
        q : int, optional
            Quantization factor
        log : bool, optional
            If ``True``, the values of the variable will be sampled
            on a logarithmic scale. Defaults to ``False``
        meta : Dict, optional
            Field for holding meta data provided by the user.
            Not used by the configuration space.
        """
        super().__init__(name=name, lower=lower, upper=upper,
                         default_value=default_value, q=q,
                         log=log, meta=meta)


class Real(CS.UniformFloatHyperparameter, Variable):
    def __init__(self, name: str, lower: Union[int, float], upper: Union[int, float],
                 default_value: Union[int, float, None] = None,
                 q: Union[int, float, None] = None, log: bool = False,
                 meta: Optional[Dict] = None) -> None:
        """
        A float variable.

        Its values are sampled from a uniform distribution with values
        from ``lower`` to ``upper``.

        Example
        -------

        >>> import openbox.utils.space as sp
        >>> cs = sp.Space(seed=1)
        >>> uniform_float = sp.Real('uni_float', lower=10, upper=100, log = False)
        >>> cs.add_variable(uniform_float)
        uni_float, Type: Real, Range: [10.0, 100.0], Default: 55.0

        Parameters
        ----------
        name : str
            Name of the variable, with which it can be accessed
        lower : int, floor
            Lower bound of a range of values from which the variable will be sampled
        upper : int, float
            Upper bound
        default_value : int, float, optional
            Sets the default value of a variable to a given value
        q : int, float, optional
            Quantization factor
        log : bool, optional
            If ``True``, the values of the variable will be sampled
            on a logarithmic scale. Default to ``False``
        meta : Dict, optional
            Field for holding meta data provided by the user.
            Not used by the configuration space.
        """
        super().__init__(name=name, lower=lower, upper=upper,
                         default_value=default_value, q=q,
                         log=log, meta=meta)


Float = Real


class Categorical(CS.CategoricalHyperparameter, Variable):
    def __init__(
            self, name: str,
            choices: Union[List[Union[str, float, int]], Tuple[Union[float, int, str]]],
            default_value: Union[int, float, str, None] = None,
            meta: Optional[Dict] = None,
            weights: Union[List[float], Tuple[float]] = None
    ) -> None:
        """
        A categorical variable.

        Its values are sampled from a set of ``values``.

        ``None`` is a forbidden value, please use a string constant instead and parse
        it in your own code, see `here <https://github.com/automl/ConfigSpace/issues/159>_`
        for further details.

        Example
        -------

        >>> import openbox.utils.space as sp
        >>> cs = sp.Space(seed=1)
        >>> categorical = sp.Categorical('cat_hp', choices=['red', 'green', 'blue'])
        >>> cs.add_variable(categorical)
        cat_hp, Type: Categorical, Choices: {red, green, blue}, Default: red

        Parameters
        ----------
        name : str
            Name of the variable, with which it can be accessed
        choices : list or tuple with str, float, int
            Collection of values to sample variable from
        default_value : int, float, str, optional
            Sets the default value of the variable to a given value
        meta : Dict, optional
            Field for holding meta data provided by the user.
            Not used by the configuration space.
        weights: (list[float], optional)
            List of weights for the choices to be used (after normalization) as probabilities during sampling, no negative values allowed
        """
        super().__init__(name=name, choices=choices, default_value=default_value,
                         meta=meta, weights=weights)


class Ordinal(CS.OrdinalHyperparameter, Variable):
    def __init__(
            self, name: str,
            sequence: Union[List[Union[float, int, str]], Tuple[Union[float, int, str]]],
            default_value: Union[str, int, float, None] = None,
            meta: Optional[Dict] = None
    ) -> None:
        """
        An ordinal variable.

        Its values are sampled form a ``sequence`` of values.
        The sequence of values from a ordinal variable is ordered.

        ``None`` is a forbidden value, please use a string constant instead and parse
        it in your own code, see `here <https://github.com/automl/ConfigSpace/issues/159>_`
        for further details.

        Example
        -------

        >>> import openbox.utils.space as sp
        >>> cs = sp.Space(seed=1)
        >>> ordinal = sp.Ordinal('ordinal_hp', sequence=['10', '20', '30'])
        >>> cs.add_variable(ordinal)
        ordinal_hp, Type: Ordinal, Sequence: {10, 20, 30}, Default: 10

        Parameters
        ----------
        name : str
            Name of the variable, with which it can be accessed.
        sequence : list or tuple with (str, float, int)
            ordered collection of values to sample variable from.
        default_value : int, float, str, optional
            Sets the default value of a variable to a given value.
        meta : Dict, optional
            Field for holding meta data provided by the user.
            Not used by the configuration space.
        """
        super().__init__(name=name, sequence=sequence, default_value=default_value, meta=meta)


class Constant(CS.Constant, Variable):
    def __init__(
            self, *args, **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)


class Space(CS.ConfigurationSpace):
    def __init__(
            self,
            name: Union[str, None] = None,
            seed: Union[int, None] = None,
            meta: Optional[Dict] = None,
            **kwargs,
    ) -> None:
        """
        A collection-like object containing a set of variable definitions and conditions.

        A configuration space organizes all variables and its conditions
        as well as its forbidden clauses. Configurations can be sampled from
        this configuration space. As underlying data structure, the
        configuration space uses a tree-based approach to represent the
        conditions and restrictions between variables.

        Parameters
        ----------
        name : str, optional
            Name of the configuration space
        seed : int, optional
            random seed
        meta : dict, optional
            Field for holding meta data provided by the user.
            Not used by the configuration space.
        """
        super().__init__(name=name, seed=seed, meta=meta, **kwargs)

    def add_variables(self, variables: List[Variable]):
        self.add_hyperparameters(variables)

    def add_variable(self, variable: Variable):
        self.add_hyperparameter(variable)


class ConditionedSpace(Space):
    """
    A configuration space that supports complex conditions between hyperparameters/variables,
        e.g., x1 <= x2 and x1 * x2 < 100.

    User can define a sample_condition function to restrict the generation of configurations.

    The following functions are guaranteed to return valid configurations:
        - self.sample_configuration()
        - get_one_exchange_neighbourhood()  # may return empty list

    Example
    -------

    >>> def sample_condition(config):
    >>>     # require x1 <= x2 and x1 * x2 < 100
    >>>     if config['x1'] > config['x2']:
    >>>         return False
    >>>     if config['x1'] * config['x2'] >= 100:
    >>>         return False
    >>>     return True
    >>>
    >>> cs = ConditionedSpace()
    >>> cs.add_variables([...])
    >>> cs.set_sample_condition(sample_condition)  # set the sample condition after all variables are added
    >>> configs = cs.sample_configuration(1000)

    """
    sample_condition: Callable[[Configuration], bool] = None

    def set_sample_condition(self, sample_condition: Callable[[Configuration], bool]):
        """
        The sample_condition function takes a configuration as input and returns a boolean value.
            - If the return value is True, the configuration is valid and will be sampled.
            - If the return value is False, the configuration is invalid and will be rejected.
        This function should be called after all hyperparameters/variables are added to the conditioned space.
        """
        self.sample_condition = sample_condition
        self._check_default_configuration()

    def _check_forbidden(self, vector: np.ndarray) -> None:
        """
        This function is called in Configuration.is_valid_configuration().
            - When Configuration.__init__() is called with values (dict), is_valid_configuration() is called.
            - When Configuration.__init__() is called with vectors (np.ndarray), there will be no validation check.
        This function is also called in get_one_exchange_neighbourhood().
        """
        # check original forbidden clauses first
        super()._check_forbidden(vector)

        if self.sample_condition is not None:
            # Populating a configuration from an array does not check if it is a legal configuration.
            # _check_forbidden() is not called. Otherwise, this would be stuck in an infinite loop.
            config = Configuration(self, vector=vector)
            if not self.sample_condition(config):
                raise ForbiddenValueError('User-defined sample condition is not satisfied.')

    def sample_configuration(self, size: int = 1) -> Union['Configuration', List['Configuration']]:
        """
        In ConfigurationSpace.sample_configuration, configurations are built with vectors (np.ndarray),
            so there will be no validation check and _check_forbidden() will not be called.
            We need to check the sample condition manually.

        Returns a single configuration if size = 1 else a list of Configurations
        """
        if self.sample_condition is None:
            return super().sample_configuration(size=size)

        if not isinstance(size, int):
            raise TypeError('Argument size must be of type int, but is %s'
                            % type(size))
        elif size < 1:
            return []

        error_iteration = 0
        accepted_configurations = []  # type: List['Configuration']
        while len(accepted_configurations) < size:
            missing = size - len(accepted_configurations)

            if missing != size:
                missing = int(1.1 * missing)
            missing += 2

            configurations = super().sample_configuration(size=missing)  # missing > 1, return a list
            configurations = [c for c in configurations if self.sample_condition(c)]
            if len(configurations) > 0:
                accepted_configurations.extend(configurations)
            else:
                error_iteration += 1
                if error_iteration > 1000:
                    raise ForbiddenValueError("Cannot sample valid configuration for %s" % self)

        if size <= 1:
            return accepted_configurations[0]
        else:
            return accepted_configurations[:size]

    def add_hyperparameter(self, *args, **kwargs):
        if self.sample_condition is not None:
            raise ValueError('Please add hyperparameter/variable before setting sample condition.')
        return super().add_hyperparameter(*args, **kwargs)

    def add_hyperparameters(self, *args, **kwargs):
        if self.sample_condition is not None:
            raise ValueError('Please add hyperparameters/variables before setting sample condition.')
        return super().add_hyperparameters(*args, **kwargs)
