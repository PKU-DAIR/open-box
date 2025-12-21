from abc import ABC
from typing import Optional, Tuple, List, Dict, TYPE_CHECKING
from openbox.utils.history import History
from openbox import logger
from ConfigSpace import ConfigurationSpace, Configuration
import json
import os
from datetime import datetime

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
                 save_compression_info: bool = False,
                 output_dir: Optional[str] = None,
                 **kwargs):
        self.origin_config_space = config_space
        self.sample_space: Optional[ConfigurationSpace] = None
        self.surrogate_space: Optional[ConfigurationSpace] = None
        self.unprojected_space: Optional[ConfigurationSpace] = None  # Target space after unprojection
        
        self.save_compression_info = save_compression_info
        self.output_dir = output_dir or './results/compression'
        self.compression_history: List[dict] = []  # Track compression updates
        
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
        
    
    def compress_space(self, 
                      space_history: Optional[List] = None,
                      source_similarities: Optional[Dict[int, float]] = None) -> Tuple[ConfigurationSpace, ConfigurationSpace]:
        if self.pipeline is not None:
            # Use pipeline mode
            self.surrogate_space, self.sample_space = self.pipeline.compress_space(
                self.origin_config_space, space_history, source_similarities
            )
            self.unprojected_space = self.pipeline.unprojected_space
            
            if self.save_compression_info:
                self._save_compression_info(event='initial_compression')
            
            return self.surrogate_space, self.sample_space
        else:
            return self._compress_space_impl(space_history)

    def get_unprojected_space(self) -> ConfigurationSpace:
        return self.pipeline.unprojected_space
    
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
                self.unprojected_space = self.pipeline.unprojected_space
                
                if self.save_compression_info:
                    self._save_compression_info(event='adaptive_update', iteration=history.num_objectives)
            
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
    
    def _save_compression_info(
        self,
        event: str = 'compression',
        iteration: Optional[int] = None
    ):
        if not self.pipeline:
            logger.warning("No pipeline configured, cannot save compression info")
            return
        
        info = {
            'timestamp': datetime.now().isoformat(),
            'event': event,
            'iteration': iteration,
            'spaces': {
                'original': {
                    'n_parameters': len(self.origin_config_space.get_hyperparameters()),
                    'parameters': self.origin_config_space.get_hyperparameter_names()
                },
                'sample': {
                    'n_parameters': len(self.sample_space.get_hyperparameters()) if self.sample_space else 0,
                    'parameters': self.sample_space.get_hyperparameter_names() if self.sample_space else []
                },
                'surrogate': {
                    'n_parameters': len(self.surrogate_space.get_hyperparameters()) if self.surrogate_space else 0,
                    'parameters': self.surrogate_space.get_hyperparameter_names() if self.surrogate_space else []
                }
            },
            'compression_ratios': {
                'sample_to_original': len(self.sample_space.get_hyperparameters()) / len(self.origin_config_space.get_hyperparameters()) if self.sample_space else 1.0,
                'surrogate_to_original': len(self.surrogate_space.get_hyperparameters()) / len(self.origin_config_space.get_hyperparameters()) if self.surrogate_space else 1.0
            },
            'pipeline': {
                'n_steps': len(self.pipeline.steps),
                'steps': []
            },
            'sampling_strategy': type(self.pipeline.get_sampling_strategy()).__name__ if self.pipeline else 'Unknown'
        }
        
        # Each step provides its own info through get_step_info()
        for i, step in enumerate(self.pipeline.steps):
            step_info = step.get_step_info()
            step_info['step_index'] = i
            info['pipeline']['steps'].append(step_info)
        
        self.compression_history.append(info)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        event_filename = f'compression_{event}_{timestamp_str}.json'
        event_filepath = os.path.join(self.output_dir, event_filename)
        
        with open(event_filepath, 'w') as f:
            json.dump(info, f, indent=2)
        logger.info(f"Saved compression info to {event_filepath}")
        
        history_filename = 'compression_history.json'
        history_filepath = os.path.join(self.output_dir, history_filename)
        
        with open(history_filepath, 'w') as f:
            json.dump({
                'total_updates': len(self.compression_history),
                'history': self.compression_history
            }, f, indent=2)
        logger.info(f"Updated compression history: {history_filepath}")
    
    def get_compression_summary(self) -> dict:
        if not self.sample_space or not self.surrogate_space:
            return {}
        
        return {
            'original_dimensions': len(self.origin_config_space.get_hyperparameters()),
            'sample_dimensions': len(self.sample_space.get_hyperparameters()),
            'surrogate_dimensions': len(self.surrogate_space.get_hyperparameters()),
            'sample_compression_ratio': len(self.sample_space.get_hyperparameters()) / len(self.origin_config_space.get_hyperparameters()),
            'surrogate_compression_ratio': len(self.surrogate_space.get_hyperparameters()) / len(self.origin_config_space.get_hyperparameters()),
            'n_updates': len(self.compression_history),
            'pipeline_steps': [step.name for step in self.pipeline.steps] if self.pipeline else []
        }

