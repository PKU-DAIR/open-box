import unittest
import numpy as np
import pytest

from openbox.surrogate.tlbo.base import BaseTLSurrogate
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.history import History, Observation
from openbox import space as sp
from openbox.utils.constants import SUCCESS, FAILED

class BaseTLSurrogateTests(unittest.TestCase):

    def setUp(self):
        self.config_space = ConfigurationSpace()
        self.config_space.add_hyperparameter(sp.Real("x1", 0, 1, default_value=0))
        # Create a History instance
        config1 = self.config_space.sample_configuration()
        config2 = self.config_space.sample_configuration()
        config3 = self.config_space.sample_configuration()

        # Set up observations with different objective values
        observation1 = Observation(config=config1, objectives=[0.1], trial_state=SUCCESS)
        observation2 = Observation(config=config2, objectives=[0.3], trial_state=SUCCESS)
        observation3 = Observation(config=config3, objectives=[0.5], trial_state=SUCCESS)

        history = History(
            num_objectives=1,
            config_space=self.config_space
        )

        # Add observations to history
        history.update_observations([observation1, observation2, observation3])

        self.source_hpo_data = [history]
        self.seed = 1
        self.history_dataset_features = None
        self.num_src_hpo_trial = 50
        self.surrogate_type = 'gp'
        self.base_surrogate = BaseTLSurrogate(self.config_space, self.source_hpo_data, self.seed,
                                              self.history_dataset_features, self.num_src_hpo_trial,
                                              self.surrogate_type)
        self.base_surrogate.build_source_surrogates()

    def test_check_base_surrogate_train(self):
        X = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        y = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        self.base_surrogate.train(X, y)
        self.assertTrue(self.base_surrogate.source_surrogates[0].is_trained)

    def test_check_base_surrogate_predict(self):
        X_train = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        y_train = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        self.base_surrogate.train(X_train, y_train)
        X_test = np.array([[0.2], [0.4], [0.6], [0.8]])
        self.base_surrogate.predict(X_test)

    def test_check_base_surrogate_predict_marginalized_over_instances(self):
        X_train = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        y_train = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        self.base_surrogate.train(X_train, y_train)
        X_test = np.array([[0.2], [0.4], [0.6], [0.8]])
        with pytest.raises(TypeError):
            self.base_surrogate.predict_marginalized_over_instances(X_test)

    def test_check_base_surrogate_combine_predictions(self):
        X_train = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        y_train = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        self.base_surrogate.train(X_train, y_train)
        X_test = np.array([[0.2], [0.4], [0.6], [0.8]])

        with pytest.raises(TypeError):
            self.base_surrogate.combine_predictions(X_test)

if __name__ == '__main__':
    unittest.main()