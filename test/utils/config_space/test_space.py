import pytest
from openbox import space as sp


def test_int_variable():
    # Define parameters for the Int variable
    name = 'uni_int'
    lower = 10
    upper = 100
    default_value = 55
    q = None
    log = False
    meta = {'description': 'Test Meta Data'}

    # Create an instance of the Int variable
    uniform_integer = sp.Int(name=name, lower=lower, upper=upper, default_value=default_value, q=q, log=log, meta=meta)

    # Check if the attributes are set correctly
    assert uniform_integer.name == name
    assert uniform_integer.lower == lower
    assert uniform_integer.upper == upper
    assert uniform_integer.default_value == default_value
    assert uniform_integer.q == q
    assert uniform_integer.log == log
    assert uniform_integer.meta == meta

    # Check if the type of the variable is correct
    assert isinstance(uniform_integer, sp.Int)
    assert isinstance(uniform_integer, sp.CS.UniformIntegerHyperparameter)
    assert isinstance(uniform_integer, sp.Variable)

    # Test string representation
    expected_repr = f"{name}, Type: UniformInteger, Range: [{lower}, {upper}], Default: {default_value}"
    assert repr(uniform_integer) == expected_repr


def test_real_variable():
    # Define parameters for the Real variable
    name = 'uni_float'
    lower = 10.0
    upper = 100.0
    default_value = 55.0
    q = None
    log = False
    meta = {'description': 'Test Meta Data'}

    # Create an instance of the Real variable
    uniform_float = sp.Real(name=name, lower=lower, upper=upper, default_value=default_value, q=q, log=log, meta=meta)

    # Check if the attributes are set correctly
    assert uniform_float.name == name
    assert uniform_float.lower == lower
    assert uniform_float.upper == upper
    assert uniform_float.default_value == default_value
    assert uniform_float.q == q
    assert uniform_float.log == log
    assert uniform_float.meta == meta

    # Check if the type of the variable is correct
    assert isinstance(uniform_float, sp.Real)
    assert isinstance(uniform_float, sp.CS.UniformFloatHyperparameter)
    assert isinstance(uniform_float, sp.Variable)

    # Test string representation
    expected_repr = f"{name}, Type: UniformFloat, Range: [{lower}, {upper}], Default: {default_value}"
    assert repr(uniform_float) == expected_repr


def test_categorical_variable():
    # Define parameters for the Categorical variable
    name = 'cat_hp'
    choices = ['red', 'green', 'blue']
    default_value = 'red'
    meta = {'description': 'Test Meta Data'}
    weights = [0.2, 0.3, 0.5]

    # Create an instance of the Categorical variable
    categorical_var = sp.Categorical(name=name, choices=choices, default_value=default_value, meta=meta,
                                     weights=weights)

    # Check if the attributes are set correctly
    assert categorical_var.name == name
    assert list(categorical_var.choices) == choices
    assert categorical_var.default_value == default_value
    assert categorical_var.meta == meta
    assert list(categorical_var.weights) == weights

    # Check if the type of the variable is correct
    assert isinstance(categorical_var, sp.Categorical)
    assert isinstance(categorical_var, sp.CS.CategoricalHyperparameter)
    assert isinstance(categorical_var, sp.Variable)

    # Test string representation
    expected_repr = f"{name}, Type: Categorical, Choices: {{{', '.join(choices)}}}, Default: {default_value}, Probabilities: (0.2, 0.3, 0.5)"
    assert repr(categorical_var) == expected_repr


def test_ordinal_variable():
    # Define parameters for the Ordinal variable
    name = 'ordinal_hp'
    sequence = ['10', '20', '30']
    default_value = '10'
    meta = {'description': 'Test Meta Data'}

    # Create an instance of the Ordinal variable
    ordinal_var = sp.Ordinal(name=name, sequence=sequence, default_value=default_value, meta=meta)

    # Check if the attributes are set correctly
    assert ordinal_var.name == name
    assert ordinal_var.sequence == tuple(sequence)
    assert ordinal_var.default_value == default_value
    assert ordinal_var.meta == meta

    # Check if the type of the variable is correct
    assert isinstance(ordinal_var, sp.Ordinal)
    assert isinstance(ordinal_var, sp.CS.OrdinalHyperparameter)
    assert isinstance(ordinal_var, sp.Variable)

    # Test string representation
    expected_repr = f"{name}, Type: Ordinal, Sequence: {{{', '.join(sequence)}}}, Default: {default_value}"
    assert repr(ordinal_var) == expected_repr


def test_space_add_variables():
    # Define variables for the Space
    int_var = sp.Int(name='int_var', lower=1, upper=10)
    real_var = sp.Real(name='real_var', lower=0.1, upper=1.0)
    categorical_var = sp.Categorical(name='cat_var', choices=['red', 'green', 'blue'])
    ordinal_var = sp.Ordinal(name='ordinal_var', sequence=['low', 'medium', 'high'])

    # Create an instance of the Space
    space = sp.Space(seed=1)

    # Add variables to the Space
    space.add_variables([int_var, real_var, categorical_var, ordinal_var])

    # Check if the variables are added correctly
    assert 'int_var' in space.get_hyperparameter_names()
    assert 'real_var' in space.get_hyperparameter_names()
    assert 'cat_var' in space.get_hyperparameter_names()
    assert 'ordinal_var' in space.get_hyperparameter_names()


def test_conditioned_space_add_variables_and_sample():
    # Define variables for the ConditionedSpace
    int_var = sp.Int(name='int_var', lower=0, upper=10, default_value=1)
    real_var = sp.Real(name='real_var', lower=0, upper=10, default_value=1.1)
    categorical_var = sp.Categorical(name='cat_var', choices=['red', 'green', 'blue'])
    ordinal_var = sp.Ordinal(name='ordinal_var', sequence=['low', 'medium', 'high'])

    # Create an instance of the ConditionedSpace
    conditioned_space = sp.ConditionedSpace(seed=1)

    # Add variables to the ConditionedSpace
    conditioned_space.add_variables([int_var, real_var, categorical_var, ordinal_var])

    # Define a sample condition function
    def sample_condition(config):
        # Require x1 (int_var) <= x2 (real_var) and x1 * x2 < 5
        if config['int_var'] > config['real_var']:
            return False
        if config['int_var'] * config['real_var'] >= 5:
            return False
        return True

    # Set the sample condition after all variables are added
    conditioned_space.set_sample_condition(sample_condition)

    # Check if the variables are added correctly
    assert 'int_var' in conditioned_space.get_hyperparameter_names()
    assert 'real_var' in conditioned_space.get_hyperparameter_names()
    assert 'cat_var' in conditioned_space.get_hyperparameter_names()
    assert 'ordinal_var' in conditioned_space.get_hyperparameter_names()

    # Test sampling configurations
    configurations = conditioned_space.sample_configuration(10)
    assert len(configurations) == 10

    # Add variables after setting conditions
    with pytest.raises(ValueError):
        conditioned_space.add_hyperparameter(int_var)

    with pytest.raises(ValueError):
        conditioned_space.add_variables([int_var, real_var, categorical_var, ordinal_var])
