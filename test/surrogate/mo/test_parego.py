import unittest
import numpy as np
from openbox.surrogate.mo.parego import ParEGOSurrogate
from openbox.surrogate.base.build_gp import create_gp_model
from openbox.surrogate.base.gp_base_prior import LognormalPrior
from openbox.surrogate.base.gp_kernels import ConstantKernel
from ConfigSpace import ConfigurationSpace


class ParEGOSurrogateTests(unittest.TestCase):

    def setUp(self):
        self.configspace = ConfigurationSpace()
        self.types = [0]
        self.bounds = [(0, 1)]
        self.seed = 1

        rng = np.random.RandomState(0)
        self.base_surrogate = create_gp_model('gp', self.configspace, self.types, self.bounds, rng)

        self.parego = ParEGOSurrogate(self.base_surrogate, seed=1)

    def test_check_parego_train(self):
        X = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        Y = np.array([[0.2, 0.4], [0.6, 0.8], [1.0, 1.2], [1.4, 1.6], [1.8, 2.0]])
        self.parego.train(X, Y)
        self.assertTrue(self.parego.base_surrogate.is_trained)

    def test_check_parego_predict(self):
        X_train = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        Y_train = np.array([[0.2, 0.4], [0.6, 0.8], [1.0, 1.2], [1.4, 1.6], [1.8, 2.0]])
        self.parego.train(X_train, Y_train)
        X_test = np.array([[0.2], [0.4], [0.6], [0.8]])
        mu, var = self.parego.predict(X_test)
        self.assertEqual(mu.shape, (4, 1))
        self.assertEqual(var.shape, (4, 1))

    def test_check_parego_predict_marginalized_over_instances(self):
        X_train = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        Y_train = np.array([[0.2, 0.4], [0.6, 0.8], [1.0, 1.2], [1.4, 1.6], [1.8, 2.0]])
        self.parego.train(X_train, Y_train)
        X_test = np.array([[0.2], [0.4], [0.6], [0.8]])
        mu, var = self.parego.predict_marginalized_over_instances(X_test)
        self.assertEqual(mu.shape, (4, 1))
        self.assertEqual(var.shape, (4, 1))

    def test_check_parego_sample_functions(self):
        X_train = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        Y_train = np.array([[0.2, 0.4], [0.6, 0.8], [1.0, 1.2], [1.4, 1.6], [1.8, 2.0]])
        self.parego.train(X_train, Y_train)
        X_test = np.array([[0.2], [0.4], [0.6], [0.8]])
        funcs = self.parego.sample_functions(X_test, n_funcs=2)
        self.assertEqual(len(funcs), 4)
        for func in funcs:
            self.assertEqual(func.shape, (2, ))


if __name__ == '__main__':
    unittest.main()