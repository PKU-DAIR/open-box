import unittest
import numpy as np
from openbox.surrogate.base.rf_with_instances_hpo import RandomForestWithInstancesHPO
from ConfigSpace import ConfigurationSpace, Configuration


class RandomForestWithInstancesHPOTests(unittest.TestCase):

    def setUp(self):
        self.types = np.array([0])
        self.bounds = [(0, 1)]
        self.rf_with_instances_hpo = RandomForestWithInstancesHPO(self.types, self.bounds)

    def test_check_rf_with_instances_hpo_train(self):
        X = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        y = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        self.rf_with_instances_hpo._train(X, y)
        self.assertIsNotNone(self.rf_with_instances_hpo.rf)

    def test_check_rf_with_instances_hpo_eval_rf(self):
        X_train = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
        y_train = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        self.rf_with_instances_hpo._train(X_train, y_train)
        X_test = np.array([[0.2], [0.4], [0.6], [0.8]])
        y_test = np.array([0.3, 0.5, 0.7, 0.9])
        cfg = self.rf_with_instances_hpo._get_configuration_space().get_default_configuration()
        loss = self.rf_with_instances_hpo._eval_rf(cfg, X_train, y_train, X_test, y_test)
        self.assertIsInstance(loss, float)

    def test_check_rf_with_instances_hpo_set_conf(self):
        cfg = self.rf_with_instances_hpo._get_configuration_space().get_default_configuration()
        rf_opts = self.rf_with_instances_hpo._set_conf(cfg, n_features=1, num_data_points=1)
        self.assertEqual(rf_opts.num_trees, cfg["num_trees"])
        self.assertEqual(rf_opts.do_bootstrapping, cfg["do_bootstrapping"])
        self.assertEqual(rf_opts.tree_opts.min_samples_to_split, int(cfg["min_samples_to_split"]))
        self.assertEqual(rf_opts.tree_opts.min_samples_in_leaf, cfg["min_samples_in_leaf"])

    def test_check_rf_with_instances_hpo_set_hypers(self):
        cfg = self.rf_with_instances_hpo._get_configuration_space().get_default_configuration()
        self.rf_with_instances_hpo._set_hypers(cfg)
        self.assertEqual(self.rf_with_instances_hpo.hypers[0], int(cfg["num_trees"]))
        self.assertEqual(self.rf_with_instances_hpo.hypers[2], cfg["do_bootstrapping"])
        self.assertEqual(self.rf_with_instances_hpo.hypers[5], cfg["min_samples_to_split"])
        self.assertEqual(self.rf_with_instances_hpo.hypers[6], cfg["min_samples_in_leaf"])

    def test_check_rf_with_instances_hpo_get_configuration_space(self):
        cfg_space = self.rf_with_instances_hpo._get_configuration_space()
        self.assertIsInstance(cfg_space, ConfigurationSpace)
        self.assertIn("num_trees", cfg_space)
        self.assertIn("do_bootstrapping", cfg_space)
        self.assertIn("max_features", cfg_space)
        self.assertIn("min_samples_to_split", cfg_space)
        self.assertIn("min_samples_in_leaf", cfg_space)

if __name__ == '__main__':
    unittest.main()