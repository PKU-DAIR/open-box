import unittest
import numpy as np
from scipy.optimize import minimize
from openbox.surrogate.tlbo.scipy_solver import Loss_func, Loss_der, scipy_solve


class TestScipySolver(unittest.TestCase):

    def setUp(self):
        self.A = np.array([[1, 2], [3, 4]])
        self.b = np.array([[5], [6]])
        self.loss_type = 0
        self.x0 = np.array([1. / self.A.shape[1]] * self.A.shape[1])

    def test_check_loss_func(self):
        pred_y = self.A * np.mat(self.x0).T
        loss = Loss_func(self.b, pred_y, self.loss_type)
        self.assertEqual(loss, 1./(self.b.shape[0])*np.linalg.norm(self.b-pred_y, 2))

    def test_check_loss_der(self):
        der = Loss_der(self.b, self.A, self.x0, self.loss_type)
        y_pred = self.A * np.mat(self.x0).T
        expected_der = -2./(self.A.shape[0])*np.array(self.A.T*(self.b-y_pred))[:, 0]
        np.testing.assert_array_equal(der, expected_der)

    def test_check_scipy_solve(self):
        x, status = scipy_solve(self.A, self.b, self.loss_type)
        self.assertTrue(status)
        self.assertFalse(np.all(x >= 0))
        self.assertFalse(np.all(x <= 1))
        self.assertTrue(sum(x) <= 1.5)


if __name__ == '__main__':
    unittest.main()