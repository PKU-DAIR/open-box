from typing import Dict, Any, Tuple, List
from ConfigSpace import ConfigurationSpace
from openbox import logger


def clip_values_to_space(values: dict, space: ConfigurationSpace, 
                        report: bool = True) -> Dict[str, Any]:
    clipped_values = values.copy()
    clipped_params = []
    
    for param_name, value in values.items():
        if param_name not in space.get_hyperparameter_names():
            # Parameter not in target space, will be filtered out elsewhere
            continue
        
        hp = space.get_hyperparameter(param_name)
        
        if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
            # Numeric parameter: clip to [lower, upper]
            original_value = value
            
            if value < hp.lower:
                clipped_values[param_name] = hp.lower
                if report:
                    clipped_params.append(
                        f"{param_name}({original_value:.4f} -> {hp.lower} [lower bound])"
                    )
            elif value > hp.upper:
                clipped_values[param_name] = hp.upper
                if report:
                    clipped_params.append(
                        f"{param_name}({original_value:.4f} -> {hp.upper} [upper bound])"
                    )
                    
        elif hasattr(hp, 'choices'):
            # Categorical parameter: check if value is valid
            if value not in hp.choices:
                # Use default value if available, otherwise first choice
                new_value = hp.default_value if hasattr(hp, 'default_value') else hp.choices[0]
                clipped_values[param_name] = new_value
                if report:
                    clipped_params.append(
                        f"{param_name}({value} -> {new_value} [invalid choice])"
                    )
    
    if clipped_params and report:
        logger.warning(f"Clipped {len(clipped_params)} parameter(s) to space bounds: {', '.join(clipped_params)}")
    
    return clipped_values


def is_within_bounds(values: dict, space: ConfigurationSpace) -> bool:
    for param_name, value in values.items():
        if param_name not in space.get_hyperparameter_names():
            continue
        
        hp = space.get_hyperparameter(param_name)
        
        if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
            if value < hp.lower or value > hp.upper:
                return False
        elif hasattr(hp, 'choices'):
            if value not in hp.choices:
                return False
    
    return True


def get_out_of_bounds_params(values: dict, space: ConfigurationSpace) -> List[str]:
    out_of_bounds = []
    
    for param_name, value in values.items():
        if param_name not in space.get_hyperparameter_names():
            continue
        
        hp = space.get_hyperparameter(param_name)
        
        if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
            if value < hp.lower or value > hp.upper:
                out_of_bounds.append(param_name)
        elif hasattr(hp, 'choices'):
            if value not in hp.choices:
                out_of_bounds.append(param_name)
    
    return out_of_bounds

