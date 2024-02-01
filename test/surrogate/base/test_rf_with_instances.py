import unittest
import numpy as np
from openbox.surrogate.base.rf_with_instances import RandomForestWithInstances


class RandomForestWithInstancesTests(unittest.TestCase):

    def setUp(self):
        self.types = np.array([0])
        self.bounds = [(0, 1)]
        self.rf_with_instances = RandomForestWithInstances(self.types, self.bounds)

    def test_check_rf_with_instances_train(self):
        X = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        y = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        self.rf_with_instances._train(X, y)
        self.assertIsNotNone(self.rf_with_instances.rf)

    def test_check_rf_with_instances_predict(self):
        X_train = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        y_train = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        self.rf_with_instances._train(X_train, y_train)
        X_test = np.array([[0.2], [0.4], [0.6], [0.8]])
        mu, var = self.rf_with_instances._predict(X_test)
        self.assertEqual(mu.shape, (4, 1))
        self.assertEqual(var.shape, (4, 1))

    def test_check_rf_with_instances_predict_marginalized_over_instances(self):
        X_train = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        y_train = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        self.rf_with_instances._train(X_train, y_train)
        X_test = np.array([[0.2], [0.4], [0.6], [0.8]])
        mu, var = self.rf_with_instances.predict_marginalized_over_instances(X_test)
        self.assertEqual(mu.shape, (4, 1))
        self.assertEqual(var.shape, (4, 1))


if __name__ == '__main__':
    unittest.main()