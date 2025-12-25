"""
Step Factory Module
Provides functions to create compression step instances from string identifiers.
"""

from typing import Optional, List, Dict, Any
from openbox import logger

from ..steps.dimension import (
    SHAPDimensionStep,
    CorrelationDimensionStep,
    ExpertDimensionStep,
    AdaptiveDimensionStep,
    SHAPImportanceCalculator,
    CorrelationImportanceCalculator,
)
from ..steps.range import (
    BoundaryRangeStep,
    SHAPBoundaryRangeStep,
    KDEBoundaryRangeStep,
    ExpertRangeStep,
)
from ..steps.projection import (
    QuantizationProjectionStep,
    REMBOProjectionStep,
    HesBOProjectionStep,
    KPCAProjectionStep,
)
from ..core import (
    CompressionStep,
    PeriodicUpdateStrategy,
    StagnationUpdateStrategy,
    ImprovementUpdateStrategy,
    HybridUpdateStrategy,
    CompositeUpdateStrategy,
)


def _create_importance_calculator_from_string(
    importance_str: Optional[str] = None,
    **kwargs
) -> Optional[Any]:
    if importance_str is None:
        return None
    
    importance_str = importance_str.lower()
    
    if importance_str in ('shap', 'shap_importance'):
        return SHAPImportanceCalculator()
    elif importance_str in ('correlation', 'corr'):
        method = kwargs.get('correlation_method', 'spearman')
        return CorrelationImportanceCalculator(method=method)
    elif importance_str == 'correlation_spearman':
        return CorrelationImportanceCalculator(method='spearman')
    elif importance_str == 'correlation_pearson':
        return CorrelationImportanceCalculator(method='pearson')
    else:
        logger.warning(
            f"Unknown importance calculator string: {importance_str}. "
            f"Using default SHAPImportanceCalculator."
        )
        return SHAPImportanceCalculator()


def _create_update_strategy_from_string(
    update_str: Optional[str] = None,
    **kwargs
) -> Optional[Any]:
    """
    Create an update strategy instance from a string identifier.
    
    Args:
        update_str: String identifier for update strategy.
                   Options: 'periodic', 'stagnation', 'improvement', 'hybrid', 'composite', 'none'
        **kwargs: Additional parameters for the strategy:
                  - For 'periodic': 'period' (default: 5)
                  - For 'stagnation': 'stagnation_threshold' (default: 5)
                  - For 'improvement': 'improvement_threshold' (default: 3)
                  - For 'hybrid': 'period', 'stagnation_threshold', 'improvement_threshold'
                  - For 'composite': expects list of strategy strings in 'composite_strategies'
    
    Returns:
        UpdateStrategy instance or None
    """
    if update_str is None or update_str.lower() == 'none':
        return None
    
    update_str = update_str.lower()
    
    if update_str == 'periodic':
        period = kwargs.get('period', 5)
        return PeriodicUpdateStrategy(period=period)
    elif update_str == 'stagnation':
        threshold = kwargs.get('stagnation_threshold', 5)
        return StagnationUpdateStrategy(threshold=threshold)
    elif update_str == 'improvement':
        threshold = kwargs.get('improvement_threshold', 3)
        return ImprovementUpdateStrategy(threshold=threshold)
    elif update_str == 'hybrid':
        period = kwargs.get('period', 10)
        stagnation_threshold = kwargs.get('stagnation_threshold', None)
        improvement_threshold = kwargs.get('improvement_threshold', None)
        return HybridUpdateStrategy(
            period=period,
            stagnation_threshold=stagnation_threshold,
            improvement_threshold=improvement_threshold
        )
    elif update_str == 'composite':
        composite_strategies = kwargs.get('composite_strategies', [])
        if not composite_strategies:
            logger.warning("No strategies provided for composite, using default periodic")
            return PeriodicUpdateStrategy(period=5)
        
        strategies = []
        for strategy_str in composite_strategies:
            strategy = _create_update_strategy_from_string(strategy_str, **kwargs)
            if strategy is not None:
                strategies.append(strategy)
        
        if not strategies:
            logger.warning("No valid strategies for composite, using default periodic")
            return PeriodicUpdateStrategy(period=5)
        
        return CompositeUpdateStrategy(*strategies)
    else:
        logger.warning(
            f"Unknown update strategy string: {update_str}. "
            f"Using default PeriodicUpdateStrategy(period=5)."
        )
        return PeriodicUpdateStrategy(period=5)


_STEP_REGISTRY = {
    'd_shap': {
        'class': SHAPDimensionStep,
        'default_params': {
            'strategy': 'shap',
            'topk': 20,
        },
        'description': 'SHAP-based dimension selection',
    },
    'd_corr': {
        'class': CorrelationDimensionStep,
        'default_params': {
            'method': 'spearman',
            'topk': 20,
        },
        'description': 'Correlation-based dimension selection',
    },
    'd_expert': {
        'class': ExpertDimensionStep,
        'default_params': {
            'strategy': 'expert',
            'expert_params': [],
        },
        'description': 'Expert knowledge-based dimension selection',
    },
    'd_adaptive': {
        'class': AdaptiveDimensionStep,
        'default_params': {
            'importance_calculator': 'shap',  # Options: 'shap', 'correlation', 'correlation_spearman', 'correlation_pearson'
            'update_strategy': 'periodic',  # Options: 'periodic', 'stagnation', 'improvement', 'hybrid', 'composite', 'none'
            'initial_topk': 30,
            'reduction_ratio': 0.2,
            'min_dimensions': 5,
            'max_dimensions': None,
        },
        'description': 'Adaptive dimension selection with dynamic topk adjustment',
    },
    'd_none': {
        'class': None,
        'default_params': {},
        'description': 'No dimension selection step',
    },
    
    'r_boundary': {
        'class': BoundaryRangeStep,
        'default_params': {
            'method': 'boundary',
            'top_ratio': 0.8,
            'sigma': 2.0,
            'enable_mixed_sampling': True,
            'initial_prob': 0.9,
        },
        'description': 'Simple boundary-based range compression',
    },
    'r_shap': {
        'class': SHAPBoundaryRangeStep,
        'default_params': {
            'method': 'shap_boundary',
            'top_ratio': 0.8,
            'sigma': 2.0,
            'enable_mixed_sampling': True,
            'initial_prob': 0.9,
        },
        'description': 'SHAP-weighted boundary range compression',
    },
    'r_kde': {
        'class': KDEBoundaryRangeStep,
        'default_params': {
            'method': 'kde_boundary',
            'source_top_ratio': 0.3,
            'kde_coverage': 0.6,
            'enable_mixed_sampling': True,
            'initial_prob': 0.9,
        },
        'description': 'KDE-based boundary range compression',
    },
    'r_expert': {
        'class': ExpertRangeStep,
        'default_params': {
            'method': 'expert',
            'expert_ranges': {},
            'enable_mixed_sampling': False,
            'initial_prob': 0.9,
        },
        'description': 'Expert-specified range compression',
    },
    'r_none': {
        'class': None,
        'default_params': {},
        'description': 'No range compression step',
    },
    
    'p_quant': {
        'class': QuantizationProjectionStep,
        'default_params': {
            'method': 'quantization',
            'max_num_values': 10,
            'seed': 42,
            'adaptive': False,
        },
        'description': 'Quantization projection step',
    },
    'p_rembo': {
        'class': REMBOProjectionStep,
        'default_params': {
            'method': 'rembo',
            'low_dim': 10,
            'seed': 42,
        },
        'description': 'REMBO projection step',
    },
    'p_hesbo': {
        'class': HesBOProjectionStep,
        'default_params': {
            'method': 'hesbo',
            'low_dim': 10,
            'max_num_values': None,
            'seed': 42,
        },
        'description': 'HesBO projection step',
    },
    'p_kpca': {
        'class': KPCAProjectionStep,
        'default_params': {
            'method': 'kpca',
            'n_components': 10,
            'kernel': 'rbf',
            'gamma': None,
            'space_history': None,
            'seed': 42,
        },
        'description': 'Kernel PCA projection step',
    },
    'p_none': {
        'class': None,
        'default_params': {},
        'description': 'No projection step',
    },
}


def get_available_step_strings() -> Dict[str, List[str]]:
    dimension_steps = [k for k in _STEP_REGISTRY.keys() if k.startswith('d_')]
    range_steps = [k for k in _STEP_REGISTRY.keys() if k.startswith('r_')]
    projection_steps = [k for k in _STEP_REGISTRY.keys() if k.startswith('p_')]
    
    return {
        'dimension': dimension_steps,
        'range': range_steps,
        'projection': projection_steps,
        'all': list(_STEP_REGISTRY.keys()),
    }


def validate_step_string(step_str: str) -> bool:
    return step_str in _STEP_REGISTRY


def get_step_info(step_str: str) -> Optional[Dict[str, Any]]:
    if not validate_step_string(step_str):
        return None
    
    info = _STEP_REGISTRY[step_str].copy()
    if info['class'] is not None:
        info['class_name'] = info['class'].__name__
    else:
        info['class_name'] = None
    info.pop('class', None)
    return info


def create_step_from_string(
    step_str: str,
    **kwargs
) -> Optional[CompressionStep]:
    """
    Create a compression step instance from a string identifier.
    
    Args:
        step_str: Step string identifier (e.g., 'd_shap', 'r_kde', 'p_quant')
        **kwargs: Additional parameters to override defaults or provide required parameters
        
    Returns:
        CompressionStep instance or None if step_str is 'd_none', 'r_none', 'p_none', or invalid.
        
    Examples:
        >>> # Create a SHAP dimension step with default parameters
        >>> step = create_step_from_string('d_shap')
        
        >>> # Create a SHAP dimension step with custom topk
        >>> step = create_step_from_string('d_shap', topk=10)
        
        >>> # Create an expert dimension step with expert parameters
        >>> step = create_step_from_string('d_expert', expert_params=['param1', 'param2'])
        
        >>> # Create a KDE range step with custom parameters
        >>> step = create_step_from_string('r_kde', source_top_ratio=0.5, kde_coverage=0.7)
        
        >>> # Create an adaptive dimension step with string-based importance and update strategy
        >>> step = create_step_from_string('d_adaptive', 
        ...                               importance_calculator='shap',
        ...                               update_strategy='periodic',
        ...                               period=10)
        
        >>> # Create an adaptive dimension step with correlation importance and stagnation strategy
        >>> step = create_step_from_string('d_adaptive',
        ...                               importance_calculator='correlation_spearman',
        ...                               update_strategy='stagnation',
        ...                               stagnation_threshold=5)
        
        >>> # Create an adaptive dimension step with hybrid update strategy
        >>> step = create_step_from_string('d_adaptive',
        ...                               importance_calculator='shap',
        ...                               update_strategy='hybrid',
        ...                               period=10,
        ...                               stagnation_threshold=5,
        ...                               improvement_threshold=3)
        
        >>> # Create a SHAP dimension step with excluded parameters
        >>> step = create_step_from_string('d_shap', 
        ...                               topk=10,
        ...                               exclude_params=['param1', 'param2'])
        
        >>> # Create a correlation dimension step with excluded parameters
        >>> step = create_step_from_string('d_corr',
        ...                               method='spearman',
        ...                               topk=15,
        ...                               exclude_params=['learning_rate', 'batch_size'])
        
        >>> # Create an expert dimension step with excluded parameters
        >>> step = create_step_from_string('d_expert',
        ...                               expert_params=['param1', 'param2', 'param3'],
        ...                               exclude_params=['param2'])  # Exclude param2 from expert selection
        
        >>> # Create an adaptive dimension step with excluded parameters
        >>> step = create_step_from_string('d_adaptive',
        ...                               importance_calculator='shap',
        ...                               update_strategy='periodic',
        ...                               exclude_params=['param1', 'param2'])
        
        >>> # Return None for 'none' steps
        >>> step = create_step_from_string('d_none')  # Returns None
    """
    if not validate_step_string(step_str):
        logger.error(f"Invalid step string identifier: {step_str}")
        logger.info(f"Available step strings: {list(_STEP_REGISTRY.keys())}")
        raise ValueError(
            f"Invalid step string identifier: {step_str}. "
            f"Available options: {list(_STEP_REGISTRY.keys())}"
        )
    
    registry_entry = _STEP_REGISTRY[step_str]
    step_class = registry_entry['class']
    
    if step_class is None:
        logger.debug(f"Step string '{step_str}' maps to None (no step)")
        return None
    
    default_params = registry_entry['default_params'].copy()
    default_params.update(kwargs)
    
    # Special handling for AdaptiveDimensionStep: convert string parameters to instances
    if step_str == 'd_adaptive' and step_class == AdaptiveDimensionStep:
        if 'importance_calculator' in default_params:
            importance_calc = default_params.pop('importance_calculator')
            if isinstance(importance_calc, str):
                importance_calculator = _create_importance_calculator_from_string(
                    importance_calc,
                    **default_params
                )
                default_params['importance_calculator'] = importance_calculator
            else:
                default_params['importance_calculator'] = importance_calc
        
        if 'update_strategy' in default_params:
            update_strategy = default_params.pop('update_strategy')
            if isinstance(update_strategy, str):
                update_strategy = _create_update_strategy_from_string(
                    update_strategy,
                    **default_params
                )
                default_params['update_strategy'] = update_strategy
            else:
                default_params['update_strategy'] = update_strategy
    
    try:
        step = step_class(**default_params)
        logger.debug(
            f"Created {step_class.__name__} from '{step_str}' "
            f"with parameters: {default_params}"
        )
        return step
    except Exception as e:
        logger.error(
            f"Failed to create step from '{step_str}': {e}. "
            f"Parameters: {default_params}"
        )
        raise


def create_steps_from_strings(
    step_strings: List[str],
    step_params: Optional[Dict[str, Dict[str, Any]]] = None
) -> List[CompressionStep]:
    """
    Create multiple compression step instances from a list of string identifiers.
    
    Args:
        step_strings: List of step string identifiers (e.g., ['d_shap', 'r_kde', 'p_quant'])
        step_params: Optional dictionary mapping step strings to their parameter dictionaries.
                     Example: {'d_shap': {'topk': 10}, 'r_kde': {'source_top_ratio': 0.5}}
    
    Returns:
        List of CompressionStep instances (None entries are filtered out for 'none' steps).
        
    Examples:
        >>> # Create steps with default parameters
        >>> steps = create_steps_from_strings(['d_shap', 'r_kde', 'p_quant'])
        
        >>> # Create steps with custom parameters
        >>> steps = create_steps_from_strings(
        ...     ['d_shap', 'r_kde'],
        ...     step_params={
        ...         'd_shap': {'topk': 10},
        ...         'r_kde': {'source_top_ratio': 0.5, 'kde_coverage': 0.7}
        ...     }
        ... )
    """
    if step_params is None:
        step_params = {}
    
    steps = []
    for step_str in step_strings:
        params = step_params.get(step_str, {})
        
        try:
            step = create_step_from_string(step_str, **params)
            if step is not None:
                steps.append(step)
            else:
                logger.debug(f"Skipping None step for '{step_str}'")
        except Exception as e:
            logger.error(f"Failed to create step '{step_str}': {e}")
            raise
    
    logger.info(f"Created {len(steps)} step(s) from {len(step_strings)} string identifier(s)")
    return steps

