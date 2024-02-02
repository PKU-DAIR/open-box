import unittest
import numpy as np
from openbox.surrogate.base.build_gp import create_gp_model
from ConfigSpace import ConfigurationSpace
from openbox.surrogate.base.gp_mcmc import GaussianProcessMCMC, GaussianProcess, GaussianProcessRegressor


class CreateGPModelTests(unittest.TestCase):

    def setUp(self):
        self.config_space = ConfigurationSpace()
        self.types = np.array([0])
        self.bounds = [(0, 1)]
        self.rng = np.random.RandomState(1)

    def test_check_gp_mcmc_model_creation(self):
        model = create_gp_model('gp_mcmc', self.config_space, self.types, self.bounds, self.rng)
        self.assertIsInstance(model, GaussianProcessMCMC)
        self.assertFalse(model.is_trained)

    def test_check_gp_model_creation(self):
        model = create_gp_model('gp', self.config_space, self.types, self.bounds, self.rng)
        self.assertIsInstance(model, GaussianProcess)
        self.assertFalse(model.is_trained)

    def test_check_gp_rbf_model_creation(self):
        model = create_gp_model('gp_rbf', self.config_space, self.types, self.bounds, self.rng)
        self.assertIsInstance(model, GaussianProcess)
        self.assertFalse(model.is_trained)

    def test_check_invalid_model_type(self):
        with self.assertRaises(ValueError):
            create_gp_model('invalid', self.config_space, self.types, self.bounds, self.rng)


if __name__ == '__main__':
    unittest.main()