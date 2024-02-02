import unittest
import numpy as np
from openbox.surrogate.tlbo.stacking_gpr import SGPR
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.history import History, Observation
from openbox import space as sp
from openbox.utils.constants import SUCCESS, FAILED


class TestSGPR(unittest.TestCase):

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
        self.sgpr = SGPR(self.config_space, self.source_hpo_data, self.seed,
                         surrogate_type=self.surrogate_type, num_src_hpo_trial=self.num_src_hpo_trial)

    def test_check_sgpr_train(self):
        X = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        y = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        self.sgpr.train(X, y)
        self.assertTrue(self.sgpr.final_regressor.is_trained)

    def test_check_sgpr_predict(self):
        X_train = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        y_train = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        self.sgpr.train(X_train, y_train)
        X_test = np.array([[0.2], [0.4], [0.6], [0.8]])
        mu, var = self.sgpr.predict(X_test)
        self.assertEqual(mu.shape, (4, 1))
        self.assertEqual(var.shape, (4, 1))

    def test_check_sgpr_get_regressor(self):
        self.sgpr.get_regressor(normalize='scale')
        self.assertTrue(self.sgpr.base_regressors[0].is_trained)

    def test_check_sgpr_train_regressor(self):
        X = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        y = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        self.sgpr.train_regressor(X, y, is_top=True)
        self.assertTrue(self.sgpr.final_regressor.is_trained)

    def test_check_sgpr_calculate_stacked_results(self):
        X_train = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        y_train = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        self.sgpr.train(X_train, y_train)
        X_test = np.array([[0.2], [0.4], [0.6], [0.8]])
        mu, var = self.sgpr.calculate_stacked_results(X_test, include_top=True)
        self.assertEqual(mu.shape, (4,))
        self.assertEqual(var.shape, (4,))


if __name__ == '__main__':
    unittest.main()