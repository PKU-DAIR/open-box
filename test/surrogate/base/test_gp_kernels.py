import unittest
import numpy as np
from unittest.mock import Mock
from openbox.surrogate.base.gp_kernels import MagicMixin, Sum, Product, ConstantKernel, Matern, RBF, WhiteKernel, HammingKernel


class TestSum(unittest.TestCase):
    def setUp(self):
        self.k1 = ConstantKernel()
        self.k2 = ConstantKernel()
        self.sum_kernel = Sum(self.k1, self.k2)

    def test_call(self):
        X = np.array([[0, 1], [1, 0]])
        K = self.sum_kernel(X)
        np.testing.assert_array_equal(K, [[2., 2.], [2., 2.]])


class TestProduct(unittest.TestCase):
    def setUp(self):
        self.k1 = ConstantKernel()
        self.k2 = ConstantKernel()
        self.product_kernel = Product(self.k1, self.k2)

    def test_call(self):
        X = np.array([[0, 1], [1, 0]])
        K = self.product_kernel(X)
        np.testing.assert_array_equal(K, [[1., 1.], [1., 1.]])


class TestConstantKernel(unittest.TestCase):
    def setUp(self):
        self.constant_kernel = ConstantKernel()

    def test_call(self):
        X = np.array([[0, 1], [1, 0]])
        K = self.constant_kernel(X)
        self.assertEqual(K.shape, (2, 2))
        self.assertTrue((K == 1.0).all())


class TestMatern(unittest.TestCase):
    def setUp(self):
        self.matern_kernel = Matern()

    def test_call(self):
        X = np.array([[0, 1], [1, 0]])
        K = self.matern_kernel(X)
        self.assertEqual(K.shape, (2, 2))


class TestRBF(unittest.TestCase):
    def setUp(self):
        self.rbf_kernel = RBF()

    def test_call(self):
        X = np.array([[0, 1], [1, 0]])
        K = self.rbf_kernel(X)
        self.assertEqual(K.shape, (2, 2))


class TestWhiteKernel(unittest.TestCase):
    def setUp(self):
        self.white_kernel = WhiteKernel()

    def test_call(self):
        X = np.array([[0, 1], [1, 0]])
        K = self.white_kernel(X)
        self.assertEqual(K.shape, (2, 2))
        self.assertTrue(np.allclose(np.diag(K), 1.0))


class TestHammingKernel(unittest.TestCase):
    def setUp(self):
        self.hamming_kernel = HammingKernel()

    def test_call(self):
        X = np.array([[0, 1], [1, 0]])
        K = self.hamming_kernel(X)
        self.assertEqual(K.shape, (2, 2))


if __name__ == '__main__':
    unittest.main()