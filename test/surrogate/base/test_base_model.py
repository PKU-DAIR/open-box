import unittest
import numpy as np
from unittest.mock import Mock
from openbox.surrogate.base.base_model import AbstractModel


class AbstractModelTests(unittest.TestCase):
    def setUp(self):
        self.types = np.array([0, 0])
        self.bounds = [(0, 1), (0, 1)]
        self.instance_features = None
        self.pca_components = 1
        self.model = AbstractModel(self.types, self.bounds, self.instance_features, self.pca_components)
        
        self.types2 = np.array([0, 0, 0, 0])
        self.instance_features2 = np.array([[0, 1], [1, 0], [1, 1], [2, 1], [4, 2]])
        self.bounds2 = [(0, 1), (0, 1), (0, 1), (0, 1)]
        self.model2 = AbstractModel(self.types2,self.bounds2, self.instance_features2, self.pca_components)

    def test_model_train_raises_not_implemented(self):

        with self.assertRaises(ValueError):
            self.model.train(np.array([1]), np.array([1]))
        with self.assertRaises(ValueError):
            self.model.train(np.array([[1]]), np.array([[1]]))
        with self.assertRaises(ValueError):
            self.model.train(np.array([[1, 2]]), np.array([[1], [1]]))

        X = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
        Y = np.array([[0], [1], [0], [1]])
        with self.assertRaises(NotImplementedError):
            self.model.train(X, Y)

    def test_model_predict_raises_not_implemented(self):
        with self.assertRaises(ValueError):
            self.model.predict(np.array([1]))
        with self.assertRaises(ValueError):
            self.model.predict(np.array([[1]]))

        X = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
        with self.assertRaises(NotImplementedError):
            self.model.predict(X)

    def test_model_predict_marginalized_over_instances(self):

        with self.assertRaises(ValueError):
            self.model.predict_marginalized_over_instances(np.array([1]))
        with self.assertRaises(ValueError):
            self.model.predict_marginalized_over_instances(np.array([[1]]))

        X = np.array([[0, 1], [3, 4], [0, 1], [1, 0]])
        with self.assertRaises(NotImplementedError):
            self.model.predict_marginalized_over_instances(X)

    def test_model_with_instances(self):
        X = np.array([[0, 1], [3, 4], [0, 1], [1, 0]])
        Y = np.array([[0], [1], [0], [1]])

        for X_, Y_ in zip(X, Y):
            X_ = np.hstack((np.tile(X_, (5,1)), self.instance_features2))
            Y_ = np.tile(Y_, (5, 1))
            with self.assertRaises(NotImplementedError):
                self.model2.train(X_, Y_)

        with self.assertRaises(NotImplementedError):
            self.model2.predict_marginalized_over_instances(X)

if __name__ == '__main__':
    unittest.main()