import copy
from typing import List, Optional, Tuple
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
    
    def compress_space(self, 
                      original_space: ConfigurationSpace,
                      space_history: Optional[List] = None) -> Tuple[ConfigurationSpace, ConfigurationSpace]:
        if self.original_space is None:
            self.original_space = original_space
        
        # Store space_history for re-compression
        self._last_space_history = space_history
        
        logger.debug(f"Starting compression pipeline with {len(self.steps)} steps")
        
        current_space = copy.deepcopy(original_space)
        current_space.seed(self.seed)
        self.space_after_steps = [current_space]
        
        for i, step in enumerate(self.steps):
            logger.debug(f"Executing step {i+1}/{len(self.steps)}: {step.name}")
            step.input_space = current_space
            current_space = step.compress(current_space, space_history)
            current_space.seed(self.seed)
            step.output_space = current_space
            self.space_after_steps.append(current_space)
            logger.debug(f"Step {i+1} output: {len(current_space.get_hyperparameters())} parameters")
        
        self._determine_spaces()
        
        self._build_sampling_strategy(original_space)
        
        logger.debug(f"Pipeline completed: sample_space={len(self.sample_space.get_hyperparameters())} params, "
                    f"surrogate_space={len(self.surrogate_space.get_hyperparameters())} params")
        logger.debug(f"Sample space: {self.sample_space}")
        logger.debug(f"Surrogate space: {self.surrogate_space}")
        logger.debug(f"Sampling strategy: {self.sampling_strategy}")
        
        return self.sample_space, self.surrogate_space
    
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
        for step in self.steps:
            # only boundary range and periodic dimension selection can support adaptive update
            if step.supports_adaptive_update():
                if step.update(self.progress, history):
                    updated = True
                    logger.info(f"Step {step.name} updated compression strategy")
        
        if updated and self.original_space is not None:
            space_history = getattr(self, '_last_space_history', None)
            if space_history is None:
                space_history = [history] if history else None
            self.compress_space(self.original_space, space_history)
            return True
        
        return False
    
    def get_sampling_strategy(self) -> SamplingStrategy:
        if self.sampling_strategy is None:
            self.sampling_strategy = StandardSamplingStrategy(self.sample_space, seed=self.seed)
        return self.sampling_strategy
    
    def needs_unproject(self) -> bool:
        return any(step.needs_unproject() for step in self.steps)
    
    def unproject_point(self, point) -> dict:
        """
        Unproject a point through all steps (in reverse order).
        
        Args:
            point: Configuration in final space
            
        Returns:
            Dictionary of configuration in original space
        """
        current_dict = point.get_dictionary() if hasattr(point, 'get_dictionary') else dict(point)

        for step in reversed(self.steps):
            if step.needs_unproject():
                from ConfigSpace import Configuration
                temp_config = Configuration(step.output_space, values=current_dict)
                current_dict = step.unproject_point(temp_config)
        return current_dict
    
    def project_point(self, point) -> dict:
        """
        Project a point through all steps (in forward order).
        
        Args:
            point: Configuration in original space
            
        Returns:
            Dictionary of configuration in final space
        """
        current_dict = point.get_dictionary() if hasattr(point, 'get_dictionary') else dict(point)
        
        for step in self.steps:
            from ConfigSpace import Configuration
            if step.input_space is not None:
                temp_config = Configuration(step.input_space, values=current_dict)
                current_dict = step.project_point(temp_config)
        return current_dict
