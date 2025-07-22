import unittest
import numpy as np
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from KNNbasic import KNearestNeighbors


class KNNbasic_tests(unittest.TestCase):

    a = np.zeros(10)
    b = np.zeros(10)
    c = np.zeros(10)
    d = np.zeros(10)

    iris = load_iris()
    X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(iris['data'], iris['target'], random_state = 3011)

    def test_K_Nearest_Neighbors_class_takes_5_arrays_pass(self):
        knn = KNearestNeighbors(self.a, self.b, self.c, self.d, 3)
        self.assertEqual(np.array_equal(knn.Xtrain, np.zeros(10)), True)
        self.assertEqual(np.array_equal(knn.Xtest, np.zeros(10)), True)
        self.assertEqual(np.array_equal(knn.ytrain, np.zeros(10)), True)
        self.assertEqual(np.array_equal(knn.ytest, np.zeros(10)), True)
        self.assertEqual(np.array_equal(knn.k, 3), True)
    
    def test_K_Nearest_Neighbors_predict_method(self):
        KNN = KNearestNeighbors(self.X_iris_train, self.y_iris_train, self.X_iris_test, self.y_iris_test, 3)
        x = KNN.Xtrain[0]
        y = KNN.ytrain[0]
        self.assertTrue(np.array_equal(KNN.predict(x, KNN.Xtrain, KNN.ytrain)[0], y))

    def test_K_Nearest_Neighbors_kpredict_method(self):
        KNN = KNearestNeighbors(self.X_iris_train, self.y_iris_train, self.X_iris_test, self.y_iris_test, 3)
        x = KNN.Xtrain[0]
        predictions = KNN.kpredict(x, KNN.k)
        self.assertTrue(len(predictions), KNN.k)
    
    def test_K_Nearest_Neighbors_final_predict_method(self):
        KNN = KNearestNeighbors(self.X_iris_train, self.y_iris_train, self.X_iris_test, self.y_iris_test, 3)
        x = KNN.Xtrain[0]
        l = KNN.Number_of_labels()
        finalPrediction = KNN.final_predict(x, KNN.k)
        prediction = KNN.kpredict(x, KNN.k)
        prediction = np.histogram(prediction, bins= l)
        self.assertTrue(np.equal(finalPrediction, np.argmax(prediction[0])))
    
    def test_K_Nearest_Neighbors_prediction_method(self):
        KNN = KNearestNeighbors(self.X_iris_train, self.y_iris_train, self.X_iris_test, self.y_iris_test, 3)
        self.assertAlmostEqual(KNN.predictions(KNN.Xtrain).all(), KNN.ytrain.all())

    def test_error_rate(self):
        NN = KNearestNeighbors(self.X_iris_train, self.y_iris_train, self.X_iris_test, self.y_iris_test, 1)
        KNN = KNearestNeighbors(self.X_iris_train, self.y_iris_train, self.X_iris_test, self.y_iris_test, 3)
        self.assertEqual(NN.error_rate(NN.Xtrain, NN.ytrain), 0)
        self.assertTrue(KNN.error_rate(KNN.Xtrain, KNN.ytrain) >= 0)
        self.assertTrue(KNN.error_rate(KNN.Xtrain, KNN.ytrain) <= 1)
        self.assertTrue(KNN.error_rate(KNN.Xtest, KNN.ytest) >= 0)
        self.assertTrue(KNN.error_rate(KNN.Xtest, KNN.ytest) <= 1)