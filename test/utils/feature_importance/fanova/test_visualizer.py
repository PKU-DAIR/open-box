import unittest
import numpy as np
from openbox.utils.feature_importance.fanova.visualizer import Visualizer
from openbox.utils.feature_importance.fanova.fanova import fANOVA
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from unittest.mock import MagicMock


class TestVisualizerInitialization(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(10, 2)
        self.Y = np.random.rand(10)
        self.config_space = ConfigurationSpace()
        self.config_space.add_hyperparameter(UniformFloatHyperparameter("x_0", 0, 1))
        self.config_space.add_hyperparameter(UniformFloatHyperparameter("x_1", 0, 1))
        self.fanova = fANOVA(self.X, self.Y, self.config_space)
        self.directory = "test/datas"

    def test_visualizer_initialization_works_correctly(self):
        visualizer = Visualizer(self.fanova, self.config_space, self.directory)
        self.assertIsNotNone(visualizer)

    def test_visualizer_raises_error_with_incorrect_directory(self):
        with self.assertRaises(FileNotFoundError):
            Visualizer(self.fanova, self.config_space, "/non/existent/directory")


class TestVisualizerCreateAllPlots(unittest.TestCase):

    def setUp(self):
        self.X = np.random.rand(10, 2)
        self.Y = np.random.rand(10)
        self.config_space = ConfigurationSpace()
        self.config_space.add_hyperparameter(UniformFloatHyperparameter("x_0", 0, 1))
        self.config_space.add_hyperparameter(UniformFloatHyperparameter("x_1", 0, 1))
        self.fanova = fANOVA(self.X, self.Y, self.config_space)
        self.directory = "test/datas"
        self.visualizer = Visualizer(self.fanova, self.config_space, self.directory)

    def test_create_all_plots_works_correctly(self):
        self.visualizer.plot_marginal = MagicMock()
        self.visualizer.plot_pairwise_marginal = MagicMock()
        self.visualizer.create_all_plots()
        self.assertTrue(self.visualizer.plot_marginal.called)
        self.assertTrue(self.visualizer.plot_pairwise_marginal.called)