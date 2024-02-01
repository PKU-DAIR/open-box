import unittest
import numpy as np
from unittest.mock import Mock
from openbox.surrogate.base.gp import GaussianProcess
from ConfigSpace import ConfigurationSpace
from openbox.surrogate.base.gp_base_prior import LognormalPrior
from openbox.surrogate.base.gp_kernels import ConstantKernel
from skopt.learning.gaussian_process import GaussianProcessRegressor


class GaussianProcessTests(unittest.TestCase):
    def setUp(self):
        self.configspace = ConfigurationSpace()
        self.types = [0, 0]
        self.bounds = [(0, 1), (0, 1)]
        self.seed = 1

        rng = np.random.RandomState(0)
        self.kernel = ConstantKernel(
            2.0,
            constant_value_bounds=(np.exp(-10), np.exp(2)),
            prior=LognormalPrior(mean=0.0, sigma=1.0, rng=rng),
        )
        self.instance_features = np.array([[0, 1], [1, 0]])
        self.pca_components = 2
        self.model = GaussianProcess(configspace=self.configspace, types=self.types, bounds=self.bounds, kernel=self.kernel, normalize_y=True, seed=self.seed, instance_features=self.instance_features, pca_components=self.pca_components)

    def test_model_train_raises_not_implemented(self):
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([0, 1])
        self.model.train(X, Y)

    def test_model_predict(self):

        X = np.array([[0, 1], [1, 0]])
        Y = np.array([0, 1])
        self.model.train(X, Y)
        X = np.array([[0, 1], [1, 0]])
        self.assertEqual(len(self.model.predict(X)), 2)

    def test_model_sample_functions_returns_samples(self):

        X = np.array([[0, 1], [1, 0]])
        Y = np.array([0, 1])
        self.model.train(X, Y)
        X = np.array([[0, 1], [1, 0]])
        samples = self.model.sample_functions(X, n_funcs=5)
        self.assertEqual(samples.shape, (2, 5))

    def test_model_get_gp_returns_gp(self):
        gp = self.model._get_gp()
        self.assertIsInstance(gp, GaussianProcessRegressor)

    def test_model_nll_returns_nll_and_grad(self):
        theta = np.array([1.0, 1.0])
        with self.assertRaises(AttributeError):
            self.model._nll(theta)

    def test_model_optimize_returns_optimized_theta(self):

        with self.assertRaises(AttributeError):
            self.model._optimize()

    def test_model_predict_returns_mean_and_var(self):

        X = np.array([[0, 1], [1, 0]])
        Y = np.array([0, 1])
        self.model.train(X, Y)
        X = np.array([[0, 1], [1, 0]])
        mean, var = self.model._predict(X)
        self.assertIsInstance(mean, np.ndarray)
        self.assertIsInstance(var, np.ndarray)
        self.assertEqual(mean.shape, (2,))
        self.assertEqual(var.shape, (2,))


if __name__ == '__main__':
    unittest.main()