import unittest
import numpy as np
from openbox.surrogate.base.gp_base_prior import TophatPrior, HorseshoePrior, LognormalPrior, SoftTopHatPrior, GammaPrior


class TestPrior(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.RandomState(0)

    def test_tophat_prior_lnprob(self):
        prior = TophatPrior(1, 10, self.rng)
        self.assertEqual(prior.lnprob(5), -np.inf)
        self.assertEqual(prior.lnprob(0), 0)
        self.assertEqual(prior.lnprob(11), -np.inf)

    def test_tophat_prior_sample_from_prior(self):
        prior = TophatPrior(1, 10, self.rng)
        samples = prior.sample_from_prior(5)
        self.assertEqual(len(samples), 5)
        self.assertFalse(all(1 <= s <= 10 for s in samples))

    def test_horseshoe_prior_lnprob(self):
        prior = HorseshoePrior(1, self.rng)
        self.assertTrue(prior.lnprob(0) > 0)
        self.assertTrue(prior.lnprob(1) < 0)

    def test_horseshoe_prior_sample_from_prior(self):
        prior = HorseshoePrior(1, self.rng)
        samples = prior.sample_from_prior(5)
        self.assertEqual(len(samples), 5)
        self.assertFalse(all(s > 0 for s in samples))

    def test_lognormal_prior_lnprob(self):
        prior = LognormalPrior(1, self.rng)
        self.assertTrue(prior.lnprob(0) < 0)
        self.assertTrue(prior.lnprob(1) < 0)

    def test_lognormal_prior_sample_from_prior(self):
        prior = LognormalPrior(1, self.rng)
        samples = prior.sample_from_prior(5)
        self.assertEqual(len(samples), 5)
        self.assertTrue(all(s > 0 for s in samples))

    def test_soft_tophat_prior_lnprob(self):
        prior = SoftTopHatPrior(1, 10, 2, self.rng)
        self.assertNotEqual(prior.lnprob(5), 0)
        self.assertFalse(prior.lnprob(0) < 0)
        self.assertTrue(prior.lnprob(11) < 0)

    def test_soft_tophat_prior_sample_from_prior(self):
        prior = SoftTopHatPrior(1, 10, 2, self.rng)
        samples = prior.sample_from_prior(5)
        self.assertEqual(len(samples), 5)
        self.assertFalse(all(1 <= s <= 10 for s in samples))

    def test_gamma_prior_lnprob(self):
        prior = GammaPrior(1, 1, 0, self.rng)
        self.assertTrue(prior.lnprob(0) < 0)
        self.assertTrue(prior.lnprob(1) < 0)

    def test_gamma_prior_sample_from_prior(self):
        prior = GammaPrior(1, 1, 0, self.rng)
        samples = prior.sample_from_prior(5)
        self.assertEqual(len(samples), 5)
        self.assertFalse(all(s > 0 for s in samples))


if __name__ == '__main__':
    unittest.main()