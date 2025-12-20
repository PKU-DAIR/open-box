import copy
from typing import List, Optional, Tuple, Dict
from openbox import logger
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace

from .step import CompressionStep
from .progress import OptimizerProgress
from ..sampling import SamplingStrategy, StandardSamplingStrategy


class CompressionPipeline:    
    def __init__(self, steps: List[CompressionStep], seed: int = 42, original_space: Optional[ConfigurationSpace] = None):
        self.steps = steps
        self.seed = seed
        self.original_space = original_space
        self.progress = OptimizerProgress()
        
        self.space_after_steps: List[ConfigurationSpace] = []
        self.sample_space: Optional[ConfigurationSpace] = None
        self.surrogate_space: Optional[ConfigurationSpace] = None
        
        self.sampling_strategy: Optional[SamplingStrategy] = None
        self.filling_strategy = None
    
    def compress_space(self, 
                      original_space: ConfigurationSpace,
                      space_history: Optional[List] = None,
                      source_similarities: Optional[Dict[int, float]] = None) -> Tuple[ConfigurationSpace, ConfigurationSpace]:
        if self.original_space is None:
            self.original_space = original_space
        
        logger.debug(f"Starting compression pipeline with {len(self.steps)} steps")
        
        current_space = copy.deepcopy(original_space)
        current_space.seed(self.seed)
        self.space_after_steps = [current_space]
        
        for i, step in enumerate(self.steps):
            input_dim = len(current_space.get_hyperparameters())
            logger.info(f"Step {i+1}/{len(self.steps)}: {step.name}")
            logger.info(f"  Input: {input_dim} parameters")
            
            step.input_space = current_space
            step.filling_strategy = self.filling_strategy
            current_space = step.compress(current_space, space_history, source_similarities)
            current_space.seed(self.seed)
            step.output_space = current_space
            
            output_dim = len(current_space.get_hyperparameters())
            dimension_ratio = output_dim / input_dim if input_dim > 0 else 1.0
            
            effective_ratio = dimension_ratio
            if hasattr(step, 'compression_info') and step.compression_info:
                if 'avg_compression_ratio' in step.compression_info:
                    effective_ratio = step.compression_info['avg_compression_ratio']
                    logger.info(f"  Output: {output_dim} parameters (dimension: {dimension_ratio:.2%}, effective: {effective_ratio:.2%})")
                else:
                    logger.info(f"  Output: {output_dim} parameters (compression ratio: {dimension_ratio:.2%})")
                logger.info(f"  Details: {step.compression_info}")
            else:
                logger.info(f"  Output: {output_dim} parameters (compression ratio: {dimension_ratio:.2%})")
            
            self.space_after_steps.append(current_space)
        
        self._determine_spaces()
        
        self._build_sampling_strategy(original_space)
        
        original_dim = len(original_space.get_hyperparameters())
        sample_dim = len(self.sample_space.get_hyperparameters())
        surrogate_dim = len(self.surrogate_space.get_hyperparameters())
        
        logger.info("=" * 60)
        logger.info("Compression Pipeline Summary")
        logger.info("=" * 60)
        logger.info(f"Original space: {original_dim} parameters")
        logger.info(f"Sample space: {sample_dim} parameters (ratio: {sample_dim/original_dim:.2%})")
        logger.info(f"Surrogate space: {surrogate_dim} parameters (ratio: {surrogate_dim/original_dim:.2%})")
        logger.info(f"Sampling strategy: {type(self.sampling_strategy).__name__}")
        logger.info("=" * 60)

        return self.surrogate_space, self.sample_space
    
    def _determine_spaces(self):
        sample_space_idx = 0
        for i, step in enumerate(self.steps):
            if step.affects_sampling_space():
                sample_space_idx = i + 1
        
        # Surrogate space is always the final output
        self.surrogate_space = self.space_after_steps[-1]
        # Sample space is determined by the last step that affects it
        self.sample_space = self.space_after_steps[sample_space_idx]
    
    def _build_sampling_strategy(self, original_space: ConfigurationSpace):
        # Check from last to first, only range compression can provide a mixed sampling strategy
        for step in reversed(self.steps):
            strategy = step.get_sampling_strategy()
            if strategy is not None:
                self.sampling_strategy = strategy
                return
        self.sampling_strategy = StandardSamplingStrategy(self.sample_space, seed=self.seed)
    
    def update_compression(self, history: History) -> bool:
        self.progress.update_from_history(history)
        
        updated = False
        updated_steps = []
        for step in self.steps:
            if step.supports_adaptive_update():
                if step.update(self.progress, history):
                    updated = True
                    updated_steps.append(step)
                    logger.info(f"Step {step.name} updated compression strategy")
        
        if updated and self.original_space is not None:
            # Check if any step needs to increase dimensions
            # For dimension increase, we must start from original_space
            # For dimension decrease, we can use progressive compression from surrogate_space
            needs_original_space = False
            for step in updated_steps:
                if hasattr(step, 'current_topk') and hasattr(step, 'initial_topk'):
                    # If current_topk > number of params in surrogate_space, need original_space
                    if self.surrogate_space and step.current_topk > len(self.surrogate_space.get_hyperparameters()):
                        needs_original_space = True
                        break
            
            if needs_original_space:
                start_space = self.original_space
                logger.debug(f"Dimension increase detected, re-compressing from original space with {len(start_space.get_hyperparameters())} parameters")
            else:
                uses_progressive = all(step.uses_progressive_compression() for step in updated_steps)
                if uses_progressive:
                    start_space = self.surrogate_space
                    logger.debug(f"Using progressive compression, starting from {len(start_space.get_hyperparameters())} parameters")
                else:
                    start_space = self.original_space
                    logger.debug(f"Using re-compression, starting from {len(start_space.get_hyperparameters())} parameters")
            
            space_history = [history] if history else None
            # Note: source_similarities should be passed from the caller (compressor/advisor)
            # Here use None since we're updating based on current task's history
            self.compress_space(start_space, space_history, source_similarities=None)
            return True
        
        return False
    
    def get_sampling_strategy(self) -> SamplingStrategy:
        if self.sampling_strategy is None:
            self.sampling_strategy = StandardSamplingStrategy(self.sample_space, seed=self.seed)
        return self.sampling_strategy
    
    def needs_unproject(self) -> bool:
        return any(step.needs_unproject() for step in self.steps)
    
    def unproject_point(self, point) -> dict:
        # Unproject a point through all steps (in reverse order)
        current_dict = point.get_dictionary() if hasattr(point, 'get_dictionary') else dict(point)

        for step in reversed(self.steps):
            if step.needs_unproject():
                current_dict = step.unproject_point(current_dict)
        return current_dict
    
    def project_point(self, point) -> dict:
        # project a point through all steps (in forward order)
        current_dict = point.get_dictionary() if hasattr(point, 'get_dictionary') else dict(point)
        
        for step in self.steps:
            if step.input_space is not None:
                current_dict = step.project_point(current_dict)
        return current_dict

