from typing import Type, Optional
from ConfigSpace import ConfigurationSpace

from .core import (
    CompressionStep,
    Compressor,
    CompressionPipeline,
    OptimizerProgress,
)

from .steps.dimension import (
    DimensionSelectionStep,
    SHAPDimensionStep,
    ExpertDimensionStep,
    CorrelationDimensionStep,
    AdaptiveDimensionStep,
)

from .steps.range import (
    RangeCompressionStep,
    BoundaryRangeStep,
    ExpertRangeStep,
    SHAPBoundaryRangeStep,
    KDEBoundaryRangeStep
)

from .steps.projection import (
    TransformativeProjectionStep,
    REMBOProjectionStep,
    HesBOProjectionStep,
    KPCAProjectionStep,
    QuantizationProjectionStep,
)

from .sampling import (
    SamplingStrategy,
    StandardSamplingStrategy,
    MixedRangeSamplingStrategy,
)

from .utils import (
    load_expert_params,
    create_space_from_ranges,
)

_COMPRESSOR_REGISTRY = {
    'pipeline': Compressor,
    'shap': None,
    'llamatune': None,
    'expert': None,
    'none': None,
}


def get_compressor(compressor_type: Optional[str] = None, 
                   config_space: ConfigurationSpace = None,
                   **kwargs):
    if compressor_type is None:
        if 'adapter_alias' in kwargs or 'le_low_dim' in kwargs:
            compressor_type = 'llamatune'
        else:
            compressor_type = kwargs.get('strategy', 'shap')
            if compressor_type == 'none':
                compressor_type = 'none'
            else:
                compressor_type = 'shap'
    
    if 'steps' in kwargs:
        steps = kwargs.pop('steps')
        return Compressor(
            config_space=config_space,
            steps=steps,
            **kwargs
        )
    
    if compressor_type == 'none':
        class NoCompressor(Compressor):
            def _compress_space_impl(self, space_history=None):
                return config_space, config_space
            def unproject_point(self, point):
                return point.get_dictionary() if hasattr(point, 'get_dictionary') else dict(point)
        return NoCompressor(config_space=config_space, **kwargs)
    
    steps = []
    
    if compressor_type == 'shap' or compressor_type == 'expert':
        strategy = kwargs.get('strategy', 'shap' if compressor_type == 'shap' else 'expert')
        
        if strategy != 'none':
            if strategy == 'expert':
                steps.append(ExpertDimensionStep(
                    strategy='expert',
                    expert_params=kwargs.get('expert_params', []),
                    expert_config_file=kwargs.get('expert_config_file', None),
                    topk=kwargs.get('topk', 20),
                ))
            else:
                steps.append(SHAPDimensionStep(
                    strategy='shap',
                    topk=kwargs.get('topk', 20),
                ))

        top_ratio = kwargs.get('top_ratio', 0.8)
        sigma = kwargs.get('sigma', 2.0)
        if top_ratio < 1.0 or sigma > 0:
            steps.append(BoundaryRangeStep(
                method='boundary',
                top_ratio=top_ratio,
                sigma=sigma,
                enable_mixed_sampling=kwargs.get('enable_mixed_sampling', True),
                initial_prob=kwargs.get('initial_prob', 0.9),
                seed=kwargs.get('seed', 42),
            ))
    
    elif compressor_type == 'pipeline':
        # Pipeline type: steps should be provided in kwargs
        # This case is already handled above, but keep for clarity
        raise ValueError("For 'pipeline' type, provide 'steps' in kwargs")
    
    elif compressor_type == 'llamatune':
        adapter_alias = kwargs.get('adapter_alias', 'none')
        le_low_dim = kwargs.get('le_low_dim', 10)
        max_num_values = kwargs.get('max_num_values', None)
        seed = kwargs.get('seed', 42)
        
        if max_num_values is not None:
            steps.append(QuantizationProjectionStep(
                method='quantization',
                max_num_values=max_num_values,
                seed=seed,
            ))
        
        if adapter_alias != 'none':
            if adapter_alias == 'rembo':
                steps.append(REMBOProjectionStep(
                    method='rembo',
                    low_dim=le_low_dim,
                    max_num_values=max_num_values,
                    seed=seed,
                ))
            elif adapter_alias == 'hesbo':
                steps.append(HesBOProjectionStep(
                    method='hesbo',
                    low_dim=le_low_dim,
                    max_num_values=max_num_values,
                    seed=seed,
                ))
            else:
                raise ValueError(f"Unknown adapter_alias: {adapter_alias}. Supported: 'rembo', 'hesbo'")
    
    else:
        raise ValueError(f"Unknown compressor type: {compressor_type}. "
                        f"Available types: {list(_COMPRESSOR_REGISTRY.keys())}")
    
    if steps:
        return Compressor(
            config_space=config_space,
            steps=steps,
            **kwargs
        )
    else:
        class NoCompressor(Compressor):
            def _compress_space_impl(self, space_history=None):
                return config_space, config_space
            def unproject_point(self, point):
                return point.get_dictionary() if hasattr(point, 'get_dictionary') else dict(point)
        return NoCompressor(config_space=config_space, **kwargs)


__all__ = [
    'CompressionStep',
    'Compressor',
    'CompressionPipeline',
    'OptimizerProgress',
    
    'DimensionSelectionStep',
    'SHAPDimensionStep',
    'ExpertDimensionStep',
    'CorrelationDimensionStep',
    'AdaptiveDimensionStep',

    'RangeCompressionStep',
    'BoundaryRangeStep',
    'ExpertRangeStep',
    'SHAPBoundaryRangeStep',
    'KDEBoundaryRangeStep',

    'TransformativeProjectionStep',
    'REMBOProjectionStep',
    'HesBOProjectionStep',
    'KPCAProjectionStep',
    'QuantizationProjectionStep',
    
    'SamplingStrategy',
    'StandardSamplingStrategy',
    'MixedRangeSamplingStrategy',
    
    'load_expert_params',
    'create_space_from_ranges',
    
    'get_compressor',
]
