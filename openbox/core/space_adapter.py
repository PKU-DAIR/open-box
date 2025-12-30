from typing import List, Optional
from ConfigSpace import Configuration, ConfigurationSpace

from openbox import logger
from openbox.utils.history import History, Observation


class IdentitySpaceAdapter:
    # Default adapter that keeps original Advisor behavior
    def __init__(self, config_space: ConfigurationSpace):
        self.original_space = config_space
        self.sample_space = config_space
        self.surrogate_space = config_space

    def setup(self, transfer_learning_history: Optional[List[History]] = None):
        return transfer_learning_history

    def config_to_surrogate(self, config: Configuration) -> Configuration:
        return config

    def config_to_sample(self, config: Configuration) -> Configuration:
        return config

    def config_to_original(self, config: Configuration) -> Configuration:
        return config

    def history_for_acq(self, history: History) -> History:
        return history

    def get_surrogate_array(self, history: History):
        return history.get_config_array(transform='scale')

    def cache_observation(self, observation: Observation):
        return

    def update(self, history: History) -> bool:
        return False


class CompressorSpaceAdapter(IdentitySpaceAdapter):
    # Adapter that bridges Advisor with the optional compressor module
    def __init__(self, config_space: ConfigurationSpace, compressor):
        super().__init__(config_space)
        self.compressor = compressor

    def setup(self, transfer_learning_history: Optional[List[History]] = None):
        source_similarities = None
        if transfer_learning_history:
            n_history = len(transfer_learning_history)
            source_similarities = {i: 1.0 / n_history for i in range(n_history)}
        surrogate_space, sample_space = self.compressor.compress_space(
            space_history=transfer_learning_history,
            source_similarities=source_similarities
        )
        self.surrogate_space = surrogate_space
        self.sample_space = sample_space
        if getattr(self.compressor, 'surrogate_space', None) is None:
            self.compressor.surrogate_space = surrogate_space
        if getattr(self.compressor, 'sample_space', None) is None:
            self.compressor.sample_space = sample_space
        return self.compressor.transform_source_data(transfer_learning_history)

    def config_to_surrogate(self, config: Configuration) -> Configuration:
        return self.compressor.convert_config_to_surrogate_space(config)

    def config_to_sample(self, config: Configuration) -> Configuration:
        convert_fn = getattr(self.compressor, 'convert_config_to_sample_space', None)
        if convert_fn is None:
            return config
        return convert_fn(config)

    def config_to_original(self, config: Configuration) -> Configuration:
        if getattr(config, 'configuration_space', None) == self.original_space:
            return config
        return self.compressor.unproject_point(config)

    def _convert_history(self, history: History, target_space: ConfigurationSpace, converter):
        converted_history = History(
            task_id=history.task_id,
            num_objectives=history.num_objectives,
            num_constraints=history.num_constraints,
            config_space=target_space,
            ref_point=history.ref_point,
            meta_info=history.meta_info,
        )
        for obs in history.observations:
            converted_obs = Observation(
                config=converter(obs.config),
                objectives=obs.objectives,
                constraints=obs.constraints,
                trial_state=obs.trial_state,
                elapsed_time=obs.elapsed_time,
                extra_info=obs.extra_info,
            )
            converted_history.update_observation(converted_obs)
        return converted_history

    def history_for_acq(self, history: History) -> History:
        if self.sample_space == self.original_space:
            return history
        return self._convert_history(
            history=history,
            target_space=self.sample_space,
            converter=self.config_to_sample,
        )

    def get_surrogate_array(self, history: History):
        if self.surrogate_space == self.original_space:
            return history.get_config_array(transform='scale')
        surrogate_history = self._convert_history(
            history=history,
            target_space=self.surrogate_space,
            converter=self.config_to_surrogate,
        )
        return surrogate_history.get_config_array(transform='scale')

    def cache_observation(self, observation: Observation):
        config = observation.config
        low_dim_cfg = getattr(config, '_low_dim_config', None)
        if low_dim_cfg is None:
            return
        if observation.extra_info is None:
            observation.extra_info = {}
        observation.extra_info['low_dim_config'] = low_dim_cfg

    def update(self, history: History) -> bool:
        updated = self.compressor.update_compression(history)
        if not updated:
            return False
        self.surrogate_space = self.compressor.surrogate_space
        self.sample_space = self.compressor.sample_space
        logger.info(
            'Compressor updated spaces: sample_dim=%d, surrogate_dim=%d',
            len(self.sample_space.get_hyperparameters()),
            len(self.surrogate_space.get_hyperparameters()),
        )
        return True
