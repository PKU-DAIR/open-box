import unittest
import numpy as np
from openbox.surrogate.base.rf_ensemble import RandomForestEnsemble
from ConfigSpace import ConfigurationSpace


class RandomForestEnsembleTests(unittest.TestCase):

    def setUp(self):
        self.types = np.array([0])
        self.bounds = [(0, 1)]
        self.s_max = 2
        self.eta = 2
        self.weight_list = [0.1, 0.2, 0.3]
        self.fusion_method = 'idp'
        self.rf_ensemble = RandomForestEnsemble(self.types, self.bounds, self.s_max, self.eta, self.weight_list, self.fusion_method)

    def test_check_rf_ensemble_train(self):
        X = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        y = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        r = 2
        self.rf_ensemble.train(X, y, r)

    def test_check_rf_ensemble_predict(self):
        X_train = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        y_train = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        for r in [1, 2, 4]:
            self.rf_ensemble.train(X_train, y_train, r)
        X_test = np.array([[0.2], [0.4], [0.6], [0.8]])
        mu, var = self.rf_ensemble._predict(X_test)
        self.assertEqual(mu.shape, (4, 1))
        self.assertEqual(var.shape, (4, 1))

    def test_check_rf_ensemble_predict_invalid_fusion(self):
        self.rf_ensemble.fusion = 'invalid'
        X_test = np.array([[0.2], [0.4], [0.6], [0.8]])
        with self.assertRaises(ValueError):
            self.rf_ensemble._predict(X_test)


if __name__ == '__main__':
    unittest.main()