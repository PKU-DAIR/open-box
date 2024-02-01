import unittest
import numpy as np
from unittest.mock import Mock
from openbox.surrogate.base.base_gp import BaseGP
from ConfigSpace import ConfigurationSpace
from skopt.learning.gaussian_process.kernels import RBF


class BaseGPTests(unittest.TestCase):
    def test_setUp(self):
        self.configspace = ConfigurationSpace()
        self.types = [0, 0]
        self.bounds = [(0, 1), (0, 1)]
        self.seed = 1
        self.kernel = RBF()
        self.instance_features = np.array([[0, 1], [1, 0]])
        self.pca_components = 2
        with self.assertRaises(NotImplementedError):
            self.model = BaseGP(self.configspace, self.types, self.bounds, self.seed, self.kernel, self.instance_features, self.pca_components)


if __name__ == '__main__':
    unittest.main()