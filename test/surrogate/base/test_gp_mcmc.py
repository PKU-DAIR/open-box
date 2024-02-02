import unittest
import numpy as np
from openbox.surrogate.base.gp_mcmc import GaussianProcessMCMC
from openbox.surrogate.base.build_gp import create_gp_model
from openbox.surrogate.base.gp_base_prior import LognormalPrior
from openbox.surrogate.base.gp_kernels import ConstantKernel
from ConfigSpace import ConfigurationSpace


class GaussianProcessMCMCTests(unittest.TestCase):

    def setUp(self):
        self.configspace = ConfigurationSpace()
        self.types = [0]
        self.bounds = [(0, 1)]
        rng = np.random.RandomState(0)
        self.kernel = ConstantKernel(
            2.0,
            constant_value_bounds=(np.exp(-10), np.exp(2)),
            prior=LognormalPrior(mean=0.0, sigma=1.0, rng=rng),
        )

        self.gp_mcmc = create_gp_model('gp_mcmc', self.configspace, self.types, self.bounds, rng)

    def test_check_train_function(self):
        X = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        y = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        self.gp_mcmc._train(X, y)
        self.assertTrue(self.gp_mcmc.is_trained)

    def test_check_predict_function(self):
        X_train = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        y_train = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        self.gp_mcmc._train(X_train, y_train)
        X_test = np.array([[0.2], [0.4], [0.6], [0.8]])
        mu, var = self.gp_mcmc._predict(X_test)
        self.assertEqual(mu.shape, (4,))
        self.assertEqual(var.shape, (4,))

    def test_check_ll_function(self):
        theta = np.array([0.1, 0.2, 0.3])
        with self.assertRaises(AttributeError):
            self.gp_mcmc._ll(theta)

    def test_check_ll_w_grad_function(self):
        theta = np.array([0.1, 0.2, 0.3])
        with self.assertRaises(AttributeError):
            self.gp_mcmc._ll_w_grad(theta)


if __name__ == '__main__':
    unittest.main()
