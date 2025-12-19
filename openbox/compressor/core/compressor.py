from abc import ABC
from typing import Optional, Tuple, List, TYPE_CHECKING
from openbox.utils.history import History
from openbox import logger
from ConfigSpace import ConfigurationSpace, Configuration

if TYPE_CHECKING:
    from ..sampling import SamplingStrategy
    from ..filling import FillingStrategy
    from .pipeline import CompressionPipeline
    from .step import CompressionStep


class Compressor(ABC):
    def __init__(self, 
                 config_space: ConfigurationSpace, 
                 filling_strategy: Optional['FillingStrategy'] = None,
                 pipeline: Optional['CompressionPipeline'] = None,
                 steps: Optional[List['CompressionStep']] = None,
                 **kwargs):
        self.origin_config_space = config_space
        self.sample_space: Optional[ConfigurationSpace] = None
        self.surrogate_space: Optional[ConfigurationSpace] = None
        self.compression_info: dict = {}
        
        if filling_strategy is None:
            from ..filling import DefaultValueFilling
            self.filling_strategy = DefaultValueFilling()
        else:
            self.filling_strategy = filling_strategy
        
        self.pipeline: Optional['CompressionPipeline'] = None
        self.seed = kwargs.get('seed', 42)
        if pipeline is not None:
            self.pipeline = pipeline
            self.pipeline.original_space = config_space
            self.pipeline.filling_strategy = self.filling_strategy
        elif steps is not None:
            from .pipeline import CompressionPipeline
            self.pipeline = CompressionPipeline(steps, seed=self.seed, original_space=config_space)
            self.pipeline.filling_strategy = self.filling_strategy
        
    
    def compress_space(self, space_history: Optional[List] = None) -> Tuple[ConfigurationSpace, ConfigurationSpace]:
        if self.pipeline is not None:
            # Use pipeline mode
            self.surrogate_space, self.sample_space = self.pipeline.compress_space(
                self.origin_config_space, space_history
            )
            self.compression_info = {
                'strategy': 'pipeline',
                'original_params': len(self.origin_config_space.get_hyperparameters()),
                'sample_params': len(self.sample_space.get_hyperparameters()),
                'surrogate_params': len(self.surrogate_space.get_hyperparameters()),
                'steps': [step.name for step in self.pipeline.steps],
            }
            return self.surrogate_space, self.sample_space
        else:
            return self._compress_space_impl(space_history)
    
    def _compress_space_impl(self, space_history: Optional[List] = None) -> Tuple[ConfigurationSpace, ConfigurationSpace]:
        raise NotImplementedError(
            "Subclasses must either provide pipeline/steps or implement _compress_space_impl"
        )
    
    def needs_unproject(self) -> bool:
        if self.pipeline is not None:
            return self.pipeline.needs_unproject()
        return False
    
    def unproject_point(self, point: Configuration) -> dict:
        if self.pipeline is not None:
            return self.pipeline.unproject_point(point)
        if hasattr(point, 'get_dictionary'):
            return point.get_dictionary()
        elif isinstance(point, dict):
            return point
        else:
            return dict(point)
    
    def project_point(self, point) -> dict:
        if self.pipeline is not None:
            return self.pipeline.project_point(point)
        if hasattr(point, 'get_dictionary'):
            return point.get_dictionary()
        elif isinstance(point, dict):
            return point
        else:
            return dict(point)
    
    def convert_config_to_surrogate_space(self, config: Configuration) -> Configuration:
        if hasattr(config, 'configuration_space') and config.configuration_space == self.surrogate_space:
            return config
        
        # project_point() handles all transformations: filtering, clipping, and filling
        projected_dict = self.project_point(config)
        
        projected_config = Configuration(self.surrogate_space, values=projected_dict)
        if hasattr(config, 'origin') and config.origin is not None:
            projected_config.origin = config.origin
        return projected_config
    
    def conver_config_to_sample_space(self, config: Configuration) -> Configuration:
        if hasattr(config, 'configuration_space') and config.configuration_space == self.sample_space:
            return config
        
        # project_point() handles all transformations: filtering, clipping, and filling
        projected_dict = self.project_point(config)
        
        sample_config = Configuration(self.sample_space, values=projected_dict)
        if hasattr(config, 'origin') and config.origin is not None:
            sample_config.origin = config.origin
        return sample_config
    
    def update_compression(self, history: History) -> bool:
        if self.pipeline is not None:
            updated = self.pipeline.update_compression(history)
            if updated:
                self.surrogate_space = self.pipeline.surrogate_space
                self.sample_space = self.pipeline.sample_space
            return updated
        return False
    
    def get_sampling_strategy(self) -> 'SamplingStrategy':
        if self.pipeline is not None:
            return self.pipeline.get_sampling_strategy()
        from ..sampling import StandardSamplingStrategy
        if self.sample_space is None:
            raise ValueError("Sample space not initialized. Call compress_space() first.")
        return StandardSamplingStrategy(self.sample_space)
    
    def transform_source_data(self, source_hpo_data: Optional[List[History]]) -> Optional[List[History]]:
        if not source_hpo_data or not self.surrogate_space:
            return source_hpo_data
        
        logger.info(f"Transforming {len(source_hpo_data)} source histories to match surrogate space")
        
        transformed = []
        for history in source_hpo_data:
            new_observations = []
            for obs in history.observations:
                new_config = self.convert_config_to_surrogate_space(obs.config)
                from openbox.utils.history import Observation
                new_obs = Observation(
                    config=new_config,
                    objectives=obs.objectives,
                    constraints=obs.constraints if hasattr(obs, 'constraints') else None,
                    trial_state=obs.trial_state if hasattr(obs, 'trial_state') else None,
                )
                new_observations.append(new_obs)
            
            new_history = History(
                task_id=history.task_id,
                num_objectives=history.num_objectives,
                num_constraints=history.num_constraints,
                config_space=self.surrogate_space,
            )
            new_history.update_observations(new_observations)
            transformed.append(new_history)
        
        logger.info(f"Successfully transformed {len(transformed)} histories")
        return transformed

