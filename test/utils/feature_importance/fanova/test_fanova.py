import unittest
import numpy as np
from openbox.utils.feature_importance.fanova.fanova import fANOVA
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter


class TestFANOVAInitialization(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(10, 2)
        self.Y = np.random.rand(10)
        self.config_space = ConfigurationSpace()
        self.config_space.add_hyperparameter(UniformFloatHyperparameter("x_0", 0, 1))
        self.config_space.add_hyperparameter(UniformFloatHyperparameter("x_1", 0, 1))

    def test_initialization_works_correctly(self):
        fanova = fANOVA(self.X, self.Y, self.config_space)
        self.assertIsNotNone(fanova.the_forest)

    def test_initialization_raises_error_with_incorrect_X(self):
        with self.assertRaises(RuntimeError):
            fANOVA(self.X[:, :1], self.Y, self.config_space)


class TestFANOVAQuantifyImportance(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(10, 2)
        self.Y = np.random.rand(10)
        self.config_space = ConfigurationSpace()
        self.config_space.add_hyperparameter(UniformFloatHyperparameter("x_0", 0, 1))
        self.config_space.add_hyperparameter(UniformFloatHyperparameter("x_1", 0, 1))

    def test_quantify_importance_returns_correct_result(self):
        fanova = fANOVA(self.X, self.Y, self.config_space)
        importance = fanova.quantify_importance((0, 1))
        self.assertIn((0, 1), importance)


class TestFANOVAGetMostImportantPairwiseMarginals(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(10, 2)
        self.Y = np.random.rand(10)
        self.config_space = ConfigurationSpace()
        self.config_space.add_hyperparameter(UniformFloatHyperparameter("x_0", 0, 1))
        self.config_space.add_hyperparameter(UniformFloatHyperparameter("x_1", 0, 1))

    def test_get_most_important_pairwise_marginals_returns_correct_result(self):
        fanova = fANOVA(self.X, self.Y, self.config_space)
        marginals = fanova.get_most_important_pairwise_marginals(n=1)
        self.assertIn(('x_0', 'x_1'), marginals)


class TestFANOVAGetTripleMarginals(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(10, 3)
        self.Y = np.random.rand(10)
        self.config_space = ConfigurationSpace()
        self.config_space.add_hyperparameter(UniformFloatHyperparameter("x_0", 0, 1))
        self.config_space.add_hyperparameter(UniformFloatHyperparameter("x_1", 0, 1))
        self.config_space.add_hyperparameter(UniformFloatHyperparameter("x_2", 0, 1))

    def test_get_triple_marginals_returns_correct_result(self):
        fanova = fANOVA(self.X, self.Y, self.config_space)
        marginals = fanova.get_triple_marginals(params=['x_0', 'x_1', 'x_2'])
        self.assertIn(('x_0', 'x_1', 'x_2'), marginals)


class TestFANOVAMarginalMeanVarianceForValues(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(10, 2)
        self.Y = np.random.rand(10)
        self.config_space = ConfigurationSpace()
        self.config_space.add_hyperparameter(UniformFloatHyperparameter("x_0", 0, 1))
        self.config_space.add_hyperparameter(UniformFloatHyperparameter("x_1", 0, 1))

    def test_marginal_mean_variance_for_values_returns_correct_result(self):
        fanova = fANOVA(self.X, self.Y, self.config_space)
        mean, variance = fanova.marginal_mean_variance_for_values([0, 1], [0.5, 0.5])
        self.assertIsNotNone(mean)
        self.assertIsNotNone(variance)


if __name__ == '__main__':
    unittest.main()