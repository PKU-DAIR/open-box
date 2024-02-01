import unittest
import numpy as np
from ConfigSpace import ConfigurationSpace
from openbox.surrogate.lightgbm import LightGBM


class LightGBMTests(unittest.TestCase):

    def setUp(self):
        self.config_space = ConfigurationSpace()
        self.types = [0]
        self.bounds = [(0, 1)]
        self.ensemble_size = 10
        self.normalize_y = True
        self.instance_features = None
        self.pca_components = None
        self.seed = 42
        self.lightgbm = LightGBM(self.config_space, self.types, self.bounds, self.ensemble_size, 
                                 self.normalize_y, self.instance_features, self.pca_components, self.seed)

    def test_check_lightgbm_train(self):
        X = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        y = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        self.lightgbm._train(X, y)
        self.assertTrue(self.lightgbm.is_trained)

    def test_check_lightgbm_predict(self):
        X_train = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        y_train = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        self.lightgbm._train(X_train, y_train)
        X_test = np.array([[0.2], [0.4], [0.6], [0.8]])
        mu, var = self.lightgbm._predict(X_test)
        self.assertEqual(mu.shape, (4,))
        self.assertEqual(var.shape, (4,))

    def test_check_lightgbm_normalize_y(self):
        y = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        normalized_y = self.lightgbm._normalize_y(y)
        self.assertAlmostEqual(np.mean(normalized_y), 0)
        self.assertAlmostEqual(np.std(normalized_y), 1)

    def test_check_lightgbm_untransform_y(self):
        y = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        normalized_y = self.lightgbm._normalize_y(y)
        untransformed_y = self.lightgbm._untransform_y(normalized_y)
        np.testing.assert_array_almost_equal(untransformed_y, y)

    def test_check_lightgbm_impute_inactive(self):
        X = np.array([[0.1], [np.nan], [0.5], [np.inf], [-np.inf]])
        imputed_X = self.lightgbm._impute_inactive(X)
        self.assertTrue(np.all(np.isfinite(imputed_X)))


if __name__ == '__main__':
    unittest.main()
