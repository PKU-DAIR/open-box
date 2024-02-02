import pytest
from openbox import space as sp
import ConfigSpace
import numpy as np

from openbox.utils.config_space.space_utils import (
    parse_bool, config_space2string,
    string2hyperparameter, string2condition, string2forbidden, string2config_space,
    get_config_values, get_config_numerical_values,
    get_config_from_dict, get_config_from_array, get_config_space_from_dict
)

from openbox.utils.config_space.util import impute_default_values, convert_configurations_to_array

from ConfigSpace import (
    Configuration, ConfigurationSpace,
    UniformIntegerHyperparameter, UniformFloatHyperparameter,
    CategoricalHyperparameter, OrdinalHyperparameter, Constant,
    EqualsCondition, InCondition,
    ForbiddenEqualsClause, ForbiddenAndConjunction, ForbiddenInClause,
)


def test_parse_bool():
    # Test case 1: Input is a bool (True)
    input_bool = True
    result_bool = parse_bool(input_bool)
    assert result_bool == input_bool

    # Test case 2: Input is a bool (False)
    input_bool_false = False
    result_bool_false = parse_bool(input_bool_false)
    assert result_bool_false == input_bool_false

    # Test case 3: Input is a string 'True'
    input_str_true = 'True'
    result_str_true = parse_bool(input_str_true)
    assert result_str_true is True

    # Test case 4: Input is a string 'False'
    input_str_false = 'False'
    result_str_false = parse_bool(input_str_false)
    assert result_str_false is False

    # Test case 5: Input is a string 'true' (case-insensitive)
    input_str_true_lower = 'true'
    result_str_true_lower = parse_bool(input_str_true_lower)
    assert result_str_true_lower is True

    # Test case 6: Input is a string 'false' (case-insensitive)
    input_str_false_lower = 'false'
    result_str_false_lower = parse_bool(input_str_false_lower)
    assert result_str_false_lower is False

    # Test case 7: Invalid string input
    input_invalid_str = 'invalid'
    with pytest.raises(ValueError, match="Expect string to be 'True' or 'False' but .* received!"):
        parse_bool(input_invalid_str)

    # Test case 8: Invalid input type
    input_invalid_type = 42
    with pytest.raises(ValueError, match="Expect a bool or str but .* received!"):
        parse_bool(input_invalid_type)


def test_config_space2string():
    # Create a ConfigurationSpace with valid hyperparameters
    cs_valid = sp.Space()
    int_var = sp.Int(name='int_var', lower=1, upper=10)
    real_var = sp.Real(name='real_var', lower=0.1, upper=1.0)
    categorical_var = sp.Categorical(name='cat_var', choices=['red', 'green', 'blue'])
    ordinal_var = sp.Ordinal(name='ordinal_var', sequence=['low', 'medium', 'high'])
    cs_valid.add_variables([int_var, real_var, categorical_var, ordinal_var])

    # Test with valid hyperparameters
    result_valid = config_space2string(cs_valid)
    assert isinstance(result_valid, str)

    # Create a ConfigurationSpace with invalid hyperparameter name
    cs_invalid_name = sp.Space()
    invalid_name_var = sp.Int(name='invalid,var', lower=1, upper=10)
    cs_invalid_name.add_variable(invalid_name_var)

    # Test with invalid hyperparameter name
    with pytest.raises(NameError, match="Invalid character in hyperparameter name!"):
        config_space2string(cs_invalid_name)

    # Create a ConfigurationSpace with invalid categorical hyperparameter value
    cs_invalid_value = sp.Space()
    invalid_value_var = sp.Categorical(name='invalid_value_var', choices=['red', 'green|blue'])
    cs_invalid_value.add_variable(invalid_value_var)

    # Test with invalid categorical hyperparameter value
    with pytest.raises(NameError, match="Invalid character in categorical hyperparameter value!"):
        config_space2string(cs_invalid_value)


def test_string2hyperparameter():
    # Test case 1: UniformFloatHyperparameter
    hp_desc_float = "    x1, Type: UniformFloat, Range: [0.1, 1.0], Default: 0.5, on log-scale, Q: 0.1"
    result_float = string2hyperparameter(hp_desc_float)
    assert isinstance(result_float, ConfigSpace.UniformFloatHyperparameter)
    assert result_float.name == 'x1'
    assert result_float.lower == 0.1
    assert result_float.upper == 1.0
    assert result_float.default_value == 0.5
    assert result_float.log is True
    assert result_float.q == 0.1

    # Test case 2: UniformIntegerHyperparameter
    hp_desc_int = "    x2, Type: UniformInteger, Range: [1, 15], Default: 4, on log-scale, Q: 2"
    result_int = string2hyperparameter(hp_desc_int)
    assert isinstance(result_int, ConfigSpace.UniformIntegerHyperparameter)
    assert result_int.name == 'x2'
    assert result_int.lower == 1
    assert result_int.upper == 15
    assert result_int.default_value == 4
    assert result_int.log is True
    assert result_int.q == 2

    # Test case 3: CategoricalHyperparameter
    hp_desc_cat = "    x3, Type: Categorical, Choices: {red, green, blue}, Default: green"
    result_cat = string2hyperparameter(hp_desc_cat)
    assert isinstance(result_cat, ConfigSpace.CategoricalHyperparameter)
    assert result_cat.name == 'x3'
    assert result_cat.choices == ('red', 'green', 'blue')
    assert result_cat.default_value == 'green'

    # Test case 4: Unsupported hyperparameter type
    hp_desc_invalid = "    x4, Type: InvalidType, Range: [1, 10], Default: 5"
    with pytest.raises(ValueError, match="Hyperparameter type .* not supported!"):
        string2hyperparameter(hp_desc_invalid)


def test_string2condition():
    # Create a ConfigurationSpace with variables
    cs = sp.Space()
    int_var = sp.Categorical(name='int_var', choices=['1', '2', '3', '5', '10'])
    cat_var = sp.Categorical(name='cat_var', choices=['red', 'green', 'blue'])
    cs.add_variables([int_var, cat_var])

    # Create a dictionary of hyperparameters
    hp_dict = {'int_var': int_var, 'cat_var': cat_var}

    # Test EqualsCondition
    cond_desc_equal = "    int_var | cat_var == 'green'"
    result_equal = string2condition(cond_desc_equal, hp_dict)
    assert isinstance(result_equal, EqualsCondition)
    assert result_equal.child.name == 'int_var'
    assert result_equal.parent.name == 'cat_var'
    assert result_equal.value == 'green'

    # Test InCondition
    cond_desc_in = "    cat_var | int_var in {'1', '5', '10'}"
    result_in = string2condition(cond_desc_in, hp_dict)
    assert isinstance(result_in, InCondition)
    assert result_in.child.name == 'cat_var'
    assert result_in.parent.name == 'int_var'
    assert result_in.values == ['1', '5', '10']

    # Test Unsupported condition type
    cond_desc_invalid = "    int_var | cat_var > 5"
    with pytest.raises(ValueError, match="Unsupported condition type in config_space!"):
        string2condition(cond_desc_invalid, hp_dict)


def test_string2forbidden():
    # Create a ConfigurationSpace with variables
    cs = sp.Space()
    int_var = sp.Categorical(name='int_var', choices=['1', '2', '3', '5', '10'])
    cat_var = sp.Categorical(name='cat_var', choices=['red', 'green', 'blue'])
    cs.add_variables([int_var, cat_var])

    # Create a dictionary of hyperparameters
    hp_dict = {'int_var': int_var, 'cat_var': cat_var}

    # Test ForbiddenEqualsClause
    forbid_desc_equal = "    Forbidden: int_var == '5'"
    result_equal = string2forbidden(forbid_desc_equal, hp_dict)
    assert isinstance(result_equal, ForbiddenEqualsClause)
    assert result_equal.hyperparameter.name == 'int_var'
    assert result_equal.value == '5'

    # Test ForbiddenInClause
    forbid_desc_in = "    (Forbidden: cat_var in {'red', 'green'})"
    result_in = string2forbidden(forbid_desc_in, hp_dict)
    assert isinstance(result_in, ForbiddenInClause)
    assert result_in.hyperparameter.name == 'cat_var'
    assert result_in.values == {'green', 'red'}

    # Test ForbiddenAndConjunction
    forbid_desc_and = "    (Forbidden: int_var == '5' && Forbidden: cat_var in {'red', 'green'})"
    result_and = string2forbidden(forbid_desc_and, hp_dict)
    assert isinstance(result_and, ForbiddenAndConjunction)
    assert isinstance(result_and.components[0], ForbiddenEqualsClause)
    assert isinstance(result_and.components[1], ForbiddenInClause)

    # Test Unsupported forbidden type
    forbid_desc_invalid = "    (Forbidden: int_var > '5')"
    with pytest.raises(ValueError, match="Unsupported forbidden type in config_space!"):
        string2forbidden(forbid_desc_invalid, hp_dict)


def test_string2config_space():
    space_desc = (
        "Configuration space object:\n"
        "  Hyperparameters:\n"
        "    x1, Type: UniformFloat, Range: [-5.0, 10.0], Default: 0.0\n"
        "    x2, Type: UniformInteger, Range: [1, 15], Default: 4, on log-scale, Q: 2\n"
        "    x3, Type: Categorical, Choices: {x1, x2, x3}, Default: x1\n"
        "    x5, Type: Categorical, Choices: {x1, x2, x3}, Default: x1\n"
        "    x6, Type: Categorical, Choices: {x1, x2, x3}, Default: x1\n"
        "    x7, Type: Categorical, Choices: {x1, x3, x2}, Default: x2\n"
        "    x9, Type: Categorical, Choices: {x1, x2, x3}, Default: x1\n"
        "  Conditions:\n"
        "    x1 | x5 == 'x1'\n"
        "    x2 | x6 in {'x1', 'x2', 'x3'}\n"
        "  Forbidden Clauses:\n"
        "    (Forbidden: x3 == 'x1' && Forbidden: x9 == 'x2' && Forbidden: x5 == 'x3')\n"
        "    Forbidden: x7 in {'x1', 'x3'}\n"
        "    Forbidden: x9 == 'x3'\n"
    )

    cs = string2config_space(space_desc)

    assert len(cs.get_hyperparameters()) == 7
    assert len(cs.get_conditions()) == 2
    assert len(cs.get_forbiddens()) == 3

    # Verify hyperparameters
    x1 = cs.get_hyperparameter('x1')
    x2 = cs.get_hyperparameter('x2')
    x3 = cs.get_hyperparameter('x3')
    assert isinstance(x1, ConfigSpace.UniformFloatHyperparameter)
    assert isinstance(x2, ConfigSpace.UniformIntegerHyperparameter)
    assert isinstance(x3, ConfigSpace.CategoricalHyperparameter)
    assert x1.lower == -5.0
    assert x1.upper == 10.0
    assert x2.lower == 1
    assert x2.upper == 15
    assert x3.choices == ('x1', 'x2', 'x3')

    # Verify conditions
    x5 = cs.get_hyperparameter('x5')
    condition = cs.get_conditions()[0]
    assert isinstance(condition, EqualsCondition)
    assert condition.child == x1
    assert condition.parent == x5
    assert condition.value == 'x1'

    # Verify forbidden clauses
    x9 = cs.get_hyperparameter('x9')
    forbidden_clause = cs.get_forbiddens()[2]
    assert isinstance(forbidden_clause, ForbiddenEqualsClause)
    assert forbidden_clause.hyperparameter == x9
    assert forbidden_clause.value == 'x3'


def test_get_config_values():
    # Create a ConfigurationSpace with variables
    cs = sp.Space()
    int_var = sp.Int(name='int_var', lower=1, upper=10)
    real_var = sp.Real(name='real_var', lower=0.1, upper=1.0)
    cat_var = sp.Categorical(name='cat_var', choices=['red', 'green', 'blue'])
    ordinal_var = sp.Ordinal(name='ordinal_var', sequence=['low', 'medium', 'high'])
    cs.add_variables([int_var, real_var, cat_var, ordinal_var])

    # Create a Configuration with values
    config_values = {'int_var': 5, 'real_var': 0.5, 'cat_var': 'green', 'ordinal_var': 'medium'}
    config = sp.Configuration(configuration_space=cs, values=config_values)

    # Test get_config_values function
    result_values = get_config_values(config)
    assert result_values == [config.get_dictionary().get(key) for key in cs.get_hyperparameter_names()]


def test_get_config_numerical_values():
    # Create a ConfigurationSpace with variables
    cs = sp.Space()
    int_var = sp.Int(name='int_var', lower=1, upper=10)
    real_var = sp.Real(name='real_var', lower=0.1, upper=1.0)
    cat_var = sp.Categorical(name='cat_var', choices=['red', 'green', 'blue'])
    ordinal_var = sp.Ordinal(name='ordinal_var', sequence=['low', 'medium', 'high'])
    cs.add_variables([int_var, real_var, cat_var, ordinal_var])

    # Create a Configuration with values
    config_values = {'int_var': 5, 'real_var': 0.5, 'cat_var': 'green', 'ordinal_var': 'medium'}
    config = sp.Configuration(configuration_space=cs, values=config_values)

    # Test get_config_numerical_values function
    result_values = get_config_numerical_values(config)
    expected_values = np.array([1, 5, 1, 0.5], dtype=float)  # 'green' -> index 1, 'medium' -> index 1

    np.testing.assert_array_equal(result_values, expected_values)


def test_get_config_from_dict():
    # Create ConfigurationSpace with hyperparameters
    cs = ConfigurationSpace()
    float_param = UniformFloatHyperparameter('float_param', lower=0.0, upper=1.0, default_value=0.5, log=False, q=0.1)
    int_param = UniformIntegerHyperparameter('int_param', lower=1, upper=10, default_value=5, log=False, q=1)
    cat_param = CategoricalHyperparameter('cat_param', choices=['red', 'green', 'blue'], default_value='red')
    const_param = Constant('const_param', 'fixed_value')
    cs.add_hyperparameters([
        float_param,
        int_param,
        cat_param,
        const_param
    ])

    # Test get_config_from_dict function
    config_dict = {
        'float_param': 0.3,
        'int_param': 8,
        'cat_param': 'green',
        'const_param': 'fixed_value'
    }
    config = get_config_from_dict(cs, config_dict)

    # Verify Configuration content
    assert isinstance(config, Configuration)
    assert config['float_param'] == 0.3
    assert config['int_param'] == 8
    assert config['cat_param'] == 'green'
    assert config['const_param'] == 'fixed_value'


def test_get_config_from_array():
    # Create ConfigurationSpace with hyperparameters
    cs = ConfigurationSpace()
    float_param = UniformFloatHyperparameter('float_param', lower=0.0, upper=1.0, default_value=0.5, log=False, q=0.1)
    int_param = UniformIntegerHyperparameter('int_param', lower=1, upper=10, default_value=5, log=False, q=1)
    cat_param = CategoricalHyperparameter('cat_param', choices=['red', 'green', 'blue'], default_value='red')
    const_param = Constant('const_param', 'fixed_value')
    cs.add_hyperparameters([
        float_param,
        int_param,
        cat_param,
        const_param
    ])

    # Test get_config_from_array function
    config_array = np.array([1, 0, 0.3, 0.8], dtype=object)
    config = get_config_from_array(cs, config_array)

    # Verify Configuration content
    assert isinstance(config, Configuration)
    assert abs(config['float_param'] - 0.3) < 1e-6
    assert config['int_param'] == 8
    assert config['cat_param'] == 'green'
    assert config['const_param'] == 'fixed_value'


def test_get_config_space_from_dict():
    # Test dictionary representation of configuration space
    space_dict = {
        'parameters': {
            'float_param': {'type': 'float', 'bound': [0.0, 1.0], 'default': 0.5, 'log': 'false', 'q': 0.1},
            'int_param': {'type': 'int', 'bound': [1, 10], 'default': 5, 'log': 'false', 'q': 1},
            'cat_param': {'type': 'cat', 'choice': ['red', 'green', 'blue'], 'default': 'red'},
            'const_param': {'type': 'const', 'value': 'fixed_value'}
        }
    }

    # Create ConfigurationSpace using get_config_space_from_dict function
    cs = get_config_space_from_dict(space_dict)

    # Verify ConfigurationSpace content
    assert len(cs.get_hyperparameters()) == 4

    # Verify float hyperparameter
    float_param = cs.get_hyperparameter('float_param')
    assert isinstance(float_param, UniformFloatHyperparameter)
    assert float_param.lower == 0.0
    assert float_param.upper == 1.0
    assert float_param.default_value == 0.5
    assert not float_param.log
    assert float_param.q == 0.1

    # Verify int hyperparameter
    int_param = cs.get_hyperparameter('int_param')
    assert isinstance(int_param, UniformIntegerHyperparameter)
    assert int_param.lower == 1
    assert int_param.upper == 10
    assert int_param.default_value == 5
    assert not int_param.log
    assert int_param.q == 1

    # Verify categorical hyperparameter
    cat_param = cs.get_hyperparameter('cat_param')
    assert isinstance(cat_param, CategoricalHyperparameter)
    assert cat_param.choices == ('red', 'green', 'blue')
    assert cat_param.default_value == 'red'

    # Verify constant hyperparameter
    const_param = cs.get_hyperparameter('const_param')
    assert isinstance(const_param, Constant)
    assert const_param.value == 'fixed_value'


def test_impute_default_values():
    # Create ConfigurationSpace with hyperparameters
    cs = sp.Space()
    float_param = cs.add_hyperparameter(
        UniformFloatHyperparameter('float_param', lower=0.0, upper=1.0, default_value=0.5, log=False, q=0.1))
    int_param = cs.add_hyperparameter(
        UniformIntegerHyperparameter('int_param', lower=1, upper=10, default_value=5, log=False, q=1))
    cat_param = cs.add_hyperparameter(
        CategoricalHyperparameter('cat_param', choices=['red', 'green', 'blue'], default_value='red'))
    const_param = cs.add_hyperparameter(Constant('const_param', 'fixed_value'))

    # Create configuration array with some inactive hyperparameters
    config_array = np.array([
        [1, 0, 0.3, 8],  # Active
        [np.nan, 0, np.nan, 5],  # Inactive float_param and cat_param
        [np.nan, 0, 0.7, 3],  # Inactive cat_param
    ])

    # Test impute_default_values function
    imputed_array = impute_default_values(cs, config_array)

    # Verify imputed configuration array
    assert np.isfinite(imputed_array[0]).all()  # No changes for the first row
    assert np.isfinite(imputed_array[1, 0])  # Imputed float_param with default
    assert imputed_array[1, 0] == cat_param.choices.index(cat_param.default_value)
    assert np.isfinite(imputed_array[1, 2])  # Imputed cat_param with default
    assert imputed_array[1, 2] == float_param.default_value
    assert np.isfinite(imputed_array[2, 2])  # Imputed cat_param with default
    assert imputed_array[2, 0] == cat_param.choices.index(cat_param.default_value)


def test_convert_configurations_to_array():
    # Create ConfigurationSpace with hyperparameters
    cs = sp.Space()
    float_param = cs.add_hyperparameter(
        UniformFloatHyperparameter('float_param', lower=0.0, upper=1.0, default_value=0.5, log=False, q=0.1))
    int_param = cs.add_hyperparameter(
        UniformIntegerHyperparameter('int_param', lower=1, upper=10, default_value=5, log=False, q=1))
    cat_param = cs.add_hyperparameter(
        CategoricalHyperparameter('cat_param', choices=['red', 'green', 'blue'], default_value='red'))
    const_param = cs.add_hyperparameter(Constant('const_param', 'fixed_value'))

    # Create list of Configuration objects with some inactive hyperparameters
    configs = [
        Configuration(cs,
                      values={'float_param': 0.3, 'int_param': 8, 'cat_param': 'green', 'const_param': 'fixed_value'}),
        # Active
        Configuration(cs,
                      values={'float_param': 0.5, 'int_param': 5, 'cat_param': 'blue', 'const_param': 'fixed_value'}),
        # Inactive float_param and cat_param
        Configuration(cs,
                      values={'float_param': 0.7, 'int_param': 3, 'cat_param': 'red', 'const_param': 'fixed_value'}),
        # Inactive cat_param
    ]

    # Test convert_configurations_to_array function
    converted_array = convert_configurations_to_array(configs)
    assert ~np.isnan(converted_array).any()
