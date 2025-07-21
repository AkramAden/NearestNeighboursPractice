import unittest
import numpy as np
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from NNbasic import NearestNeighbors

class NNbasic_tests(unittest.TestCase):

    a = np.zeros(10)
    b = np.zeros(10)
    c = np.zeros(10)
    d = np.zeros(10)
    nn = NearestNeighbors(a, b, c, d)

    iris = load_iris()
    X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(iris['data'], iris['target'], random_state = 3011)

    NN = NearestNeighbors(X_iris_train, y_iris_train, X_iris_test, y_iris_test)

    def test_Nearest_Neighbors_class_takes_4_arrays_pass(self):
        self.assertEqual(np.array_equal(self.nn.Xtrain, np.zeros(10)), True)
        self.assertEqual(np.array_equal(self.nn.Xtest, np.zeros(10)), True)
        self.assertEqual(np.array_equal(self.nn.ytrain, np.zeros(10)), True)
        self.assertEqual(np.array_equal(self.nn.ytest, np.zeros(10)), True)
    
    def test_Nearest_Neighbors_E_len_method(self):
        v = [3, 3 , 3, 3]
        e = math.sqrt(9 + 9 + 9 + 9)
        self.assertEqual(self.nn.E_len(v), e)

    def test_Nearest_Neighbors_predict_method_pass(self):
        x = self.NN.Xtrain[0]
        y = self.NN.ytrain[0]
        self.assertTrue(self.NN.predict(x) == y)
    
    def test_Nearest_Neighbors_predict_method_fail(self):
        x = self.NN.Xtrain[0]
        y = self.NN.ytrain[0]
        self.assertFalse(self.NN.predict(x) == y + 1)

    def test_Nearest_Neighbors_prediction_method(self):
        self.assertTrue(np.array_equal(self.NN.predictions(self.NN.Xtrain).all(), self.NN.ytrain.all()))
    
    def test_error_rate(self):
        self.assertEqual(self.NN.error_rate(self.NN.Xtrain, self.NN.ytrain), 0)
        self.assertTrue(self.NN.error_rate(self.NN.Xtest, self.NN.ytest) >= 0)
        self.assertTrue(self.NN.error_rate(self.NN.Xtest, self.NN.ytest) <= 1)
        

