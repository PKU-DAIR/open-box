# License: MIT

import re
import numpy as np
from ConfigSpace import (
    Configuration, ConfigurationSpace,
    UniformIntegerHyperparameter, UniformFloatHyperparameter,
    CategoricalHyperparameter, OrdinalHyperparameter, Constant,
    EqualsCondition, InCondition,
    ForbiddenEqualsClause, ForbiddenAndConjunction, ForbiddenInClause,
)


def parse_bool(input_):
    if isinstance(input_, bool):
        return input
    elif isinstance(input_, str):
        if input_.lower == 'true':
            return True
        elif input_.lower() == 'false':
            return False
        else:
            raise ValueError("Expect string to be 'True' or 'False' but %s received!" % input_)
    else:
        ValueError("Expect a bool or str but %s received!" % type(input_))


def config_space2string(config_space: ConfigurationSpace):
    pattern = r'[,|{}\'=<>&]'
    for hp in config_space.get_hyperparameters():
        if re.search(pattern, hp.name):
            raise NameError('Invalid character in hyperparameter name!')
        if hasattr(hp, 'choices'):
            for value in hp.choices:
                if re.search(pattern, value):
                    raise NameError('Invalid character in categorical hyperparameter value!')
    return str(config_space)


def string2hyperparameter(hp_desc: str):
    # Only support type, range, default_value, log, q
    # Sample: x2, Type: UniformInteger, Range: [1, 15], Default: 4, on log-scale, Q: 2
    q = -1
    log = None
    default_value = None
    params = hp_desc.split(',')
    cur_idx = -1
    while default_value is None:
        if q == -1:
            if 'Q:' in params[cur_idx]:
                q = float(params[cur_idx][4:])
                cur_idx -= 1
                continue
            else:
                q = None
        if log is None:
            if 'log-scale' in params[cur_idx]:
                log = True
                cur_idx -= 1
                continue
            else:
                log = False
        if default_value is None:
            default_value = str(params[cur_idx][10:])
            cur_idx -= 1

    prefix_params = ','.join(params[:cur_idx + 1])
    range_str = prefix_params.split(':')[-1]
    if range_str[-1] == ']':
        element_list = range_str[2:-1].split(',')
        range = [float(element_list[0]), float(element_list[1])]
    else:
        element_list = range_str[1:-1].split(',')
        range = [element[1:] for element in element_list]

    type_str = prefix_params.split(':')[-2].split(',')[0][1:]

    name_str = ':'.join(prefix_params.split(':')[:-2])
    name = ','.join(name_str.split(',')[:-1])[4:]

    if type_str == 'UniformFloat':
        return UniformFloatHyperparameter(name, range[0], range[1], default_value=float(default_value), log=log, q=q)
    elif type_str == 'UniformInteger':
        return UniformIntegerHyperparameter(name, range[0], range[1], default_value=int(default_value), log=log, q=q)
    elif type_str == 'Categorical':
        return CategoricalHyperparameter(name, range, default_value=default_value)
    else:
        raise ValueError('Hyperparameter type %s not supported!' % type)


def string2condition(cond_desc: str, hp_dict: dict):
    # Support EqualCondition and InCondition
    pattern_in = r'(.*?)\sin\s(.*?)}'
    pattern_equal = r'(.*?)\s==\s(.*)'
    matchobj_equal = re.match(pattern_equal, cond_desc)
    matchobj_in = re.match(pattern_in, cond_desc)
    if matchobj_equal:
        two_elements = matchobj_equal.group(1).split('|')
        child_name = two_elements[0][4:-1]
        parent_name = two_elements[1][1:]
        target_value = matchobj_equal.group(2)[1:-1]
        cond = EqualsCondition(hp_dict[child_name], hp_dict[parent_name], target_value)
    elif matchobj_in:
        two_elements = matchobj_in.group(1).split('|')
        child_name = two_elements[0][4:-1]
        parent_name = two_elements[1][1:]
        choice_str = matchobj_in.group(2).split(',')
        choices = [choice[2:-1] for choice in choice_str]
        cond = InCondition(hp_dict[child_name], hp_dict[parent_name], choices)
    else:
        raise ValueError("Unsupported condition type in config_space!")
    return cond


def string2forbidden(forbid_desc: str, hp_dict: dict):
    def string2forbidden_base(base_forbid_desc: str, hp_dict: dict):
        pattern_equal = r'[\s(]*Forbidden:\s(.*?)\s==\s(.*)'
        pattern_in = r'[\s(]*Forbidden:\s(.*?)\sin\s(.*)?}'
        matchobj_equal = re.match(pattern_equal, base_forbid_desc)
        matchobj_in = re.match(pattern_in, base_forbid_desc)
        if matchobj_equal:
            forbid_name = matchobj_equal.group(1)
            target_value = matchobj_equal.group(2)[1:-1]
            forbid = ForbiddenEqualsClause(hp_dict[forbid_name], target_value)
        elif matchobj_in:
            forbid_name = matchobj_in.group(1)
            choice_str = matchobj_in.group(2).split(',')
            choices = [choice[2:-1] for choice in choice_str]
            forbid = ForbiddenInClause(hp_dict[forbid_name], choices)
        else:
            raise ValueError("Unsupported forbidden type in config_space!")
        return forbid

    forbidden_strlist = forbid_desc.split('&&')
    if len(forbidden_strlist) == 1:
        return string2forbidden_base(forbid_desc, hp_dict)
    else:
        forbiddden_list = [string2forbidden_base(split_forbidden[:-1], hp_dict) for split_forbidden in
                           forbidden_strlist]
        return ForbiddenAndConjunction(*forbiddden_list)


def string2config_space(space_desc: str):
    line_list = space_desc.split('\n')
    cur_line = 2
    cs = ConfigurationSpace()
    status = 'hp'
    hp_list = list()
    while cur_line != len(line_list) - 1:
        line_content = line_list[cur_line]
        if line_content == '  Conditions:':
            hp_dict = {hp.name: hp for hp in hp_list}
            status = 'cond'
        elif line_content == '  Forbidden Clauses:':
            status = 'bid'
        else:
            if status == 'hp':
                hp = string2hyperparameter(line_content)
                hp_list.append(hp)
                cs.add_hyperparameter(hp)
            elif status == 'cond':
                cond = string2condition(line_content, hp_dict)
                cs.add_condition(cond)
            else:
                forbid = string2forbidden(line_content, hp_dict)
                cs.add_forbidden_clause(forbid)
        cur_line += 1
    return cs


def get_config_values(config: Configuration):
    # DO NOT USE config.get_dictionary().values()! may get random value order for different configs
    config_space = config.configuration_space
    config_values = [config.get_dictionary().get(key) for key in config_space.get_hyperparameter_names()]
    return config_values


def get_config_numerical_values(config: Configuration):
    """
    Get the numerical values of a configuration.
    For categorical hyperparameters, the index of the value in the choices list is used.
    For numerical hyperparameters, the value is used.
    """
    X_from_dict = np.array(get_config_values(config), dtype=object)
    X_from_array = config.get_array()
    discrete_types = (CategoricalHyperparameter, OrdinalHyperparameter, Constant)
    config_space = config.configuration_space
    discrete_idx = [isinstance(hp, discrete_types) for hp in config_space.get_hyperparameters()]
    X = X_from_dict.copy()
    X[discrete_idx] = X_from_array[discrete_idx]
    X = X.astype(X_from_array.dtype)
    return X


def round_config(config: Configuration):
    """
    Round config if q is set in Int/Float hyperparameter.
    Make config.get_array() return the correct value.
    """
    return Configuration(configuration_space=config.configuration_space, values=config.get_dictionary())


def get_config_from_dict(config_space: ConfigurationSpace, config_dict: dict):
    # update:
    #   inactive_with_values is not allowed, or you should use ConfigSpace.util.deactivate_inactive_hyperparameters
    config = Configuration(configuration_space=config_space, values=config_dict)
    return round_config(config)


def get_config_from_array(config_space: ConfigurationSpace, config_array: np.ndarray):
    config = Configuration(configuration_space=config_space, vector=config_array)
    return round_config(config)


def get_config_space_from_dict(space_dict: dict):
    cs = ConfigurationSpace()
    params_dict = space_dict['parameters']
    for key in params_dict:
        param_dict = params_dict[key]
        param_type = param_dict['type']
        if param_type in ['float', 'real', 'int', 'integer']:
            bound = param_dict['bound']
            optional_args = dict()
            if 'default' in param_dict:
                optional_args['default_value'] = param_dict['default']
            if 'log' in param_dict:
                optional_args['log'] = parse_bool(param_dict['log'])
            if 'q' in param_dict:
                optional_args['q'] = param_dict['q']

            if param_type in ['float', 'real']:
                param = UniformFloatHyperparameter(key, bound[0], bound[1], **optional_args)
            else:
                param = UniformIntegerHyperparameter(key, bound[0], bound[1], **optional_args)

        elif param_type in ['cat', 'cate', 'categorical']:
            choices = param_dict['choice']
            optional_args = dict()
            if 'default' in param_dict:
                optional_args['default_value'] = param_dict['default']
            param = CategoricalHyperparameter(key, choices, **optional_args)

        elif param_type in ['const', 'constant']:
            value = param_dict['value']
            param = Constant(key, value)

        else:
            raise ValueError("Parameter type %s not supported!" % param_type)

        cs.add_hyperparameter(param)
    return cs
