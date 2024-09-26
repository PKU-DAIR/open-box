import unittest
import numpy as np
from openbox.acq_optimizer.random_configuration_chooser import ChooserNoCoolDown, ChooserLinearCoolDown, ChooserProb, ChooserProbCoolDown, ChooserCosineAnnealing


class TestRandomConfigurationChooser(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.RandomState(1)

    def test_no_cool_down_chooser_returns_expected_result(self):
        chooser = ChooserNoCoolDown(2.0)
        self.assertTrue(chooser.check(2))
        self.assertFalse(chooser.check(3))

    def test_linear_cool_down_chooser_returns_expected_result(self):
        chooser = ChooserLinearCoolDown(2.0, 0.3, np.inf)
        self.assertTrue(chooser.check(2))
        self.assertFalse(chooser.check(3))

    def test_prob_chooser_returns_expected_result(self):
        chooser = ChooserProb(0.5, self.rng)
        self.assertTrue(chooser.check(1))
        self.assertFalse(chooser.check(2))

    def test_prob_cool_down_chooser_returns_expected_result(self):
        chooser = ChooserProbCoolDown(0.5, 0.1, self.rng)
        self.assertTrue(chooser.check(1))
        self.assertFalse(chooser.check(2))

    def test_cosine_annealing_chooser_returns_expected_result(self):
        chooser = ChooserCosineAnnealing(0.5, 0.1, 2, self.rng)
        self.assertTrue(chooser.check(1))
        self.assertFalse(chooser.check(2))


if __name__ == '__main__':
    unittest.main()