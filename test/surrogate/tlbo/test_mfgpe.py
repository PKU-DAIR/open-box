import unittest
import numpy as np
from openbox.surrogate.tlbo.mfgpe import MFGPE
from openbox.utils.history import History, Observation
from openbox import space as sp
from openbox.utils.constants import SUCCESS, FAILED
from openbox.utils.config_space import ConfigurationSpace


class MFGPESurrogateTests(unittest.TestCase):

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

        self.source_hpo_data = [history, history]
        self.seed = 1
        self.surrogate_type = 'gp'
        self.num_src_hpo_trial = 50
        self.mfgpe = MFGPE(self.config_space, self.source_hpo_data, self.seed,
                           surrogate_type=self.surrogate_type, num_src_hpo_trial=self.num_src_hpo_trial)

    def test_check_mfgpe_train(self):
        X = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        y = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        self.mfgpe.train(X, y)
        self.assertTrue(self.mfgpe.source_surrogates[0].is_trained)

    def test_check_mfgpe_predict(self):
        X_train = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        y_train = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        self.mfgpe.train(X_train, y_train)
        X_test = np.array([[0.2], [0.4], [0.6], [0.8]])
        mu, var = self.mfgpe.predict(X_test)
        self.assertEqual(mu.shape, (4, 1))
        self.assertEqual(var.shape, (4, 1))

    def test_check_mfgpe_update_mf_trials(self):
        config_space = ConfigurationSpace()
        config_space.add_hyperparameter(sp.Real("x1", 0, 1, default_value=0))
        # Create a History instance
        config1 = config_space.sample_configuration()
        config2 = config_space.sample_configuration()

        # Set up observations with different objective values
        observation1 = Observation(config=config1, objectives=[0.1], trial_state=SUCCESS)
        observation2 = Observation(config=config2, objectives=[0.3], trial_state=SUCCESS)

        history = History(
            num_objectives=1,
            config_space=config_space
        )

        # Add observations to history
        history.update_observations([observation1, observation2])

        mf_hpo_data = [history, history]
        self.mfgpe.update_mf_trials(mf_hpo_data)
        self.assertEqual(self.mfgpe.K, len(mf_hpo_data) - 1)
        self.assertEqual(self.mfgpe.w, [1. / self.mfgpe.K] * self.mfgpe.K + [0.])

    def test_check_mfgpe_get_weights(self):
        weights = self.mfgpe.get_weights()
        self.assertEqual(weights, self.mfgpe.w)


if __name__ == '__main__':
    unittest.main()