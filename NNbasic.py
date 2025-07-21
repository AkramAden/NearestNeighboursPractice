import numpy as np
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()

X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(iris['data'], iris['target'], random_state = 3011)
print(X_iris_train.shape, X_iris_test.shape, y_iris_train.shape, y_iris_test.shape)


class NearestNeighbors:

    # Here is the initalization method of the NearestNeighbors class. It takes the 4 array arguments and saves
    # them as its own private variables, as shown below. 
    
    def __init__(self, Xtrain, ytrain, Xtest, ytest):
        self.Xtrain = Xtrain                          
        # Xtrain is the array that contains the data of the training set of the dataset.
        self.ytrain = ytrain                          
        # ytrain is the array that contains the true labels of the training set of the dataset.
        self.Xtest = Xtest                            
        # Xtest is the array that contains the data of the test set of the dataset.
        self.ytest = ytest                            
        # ytrain is the array that contains the true labels of the test set of the dataset.
        self.ypred = ""
        # ypred is an array that will be used to hold the predicted labels of each element of a given test set.

    #____________________________________________________________________________________________________________________________

    # Here is the E_len() method of the NearestNeighbors class.
    # It takes a single argument, x, as an argument and calulates and returns its L2/Euclidean norm, i.e. distance from the origin.

    def E_len(self, x):
        return np.linalg.norm(x)
    
    #____________________________________________________________________________________________________________________________


    # Here is the predict(x) method of the NearestNeighbors class.
    # It takes a single argument, 'x', and through calculates its nearest neighbour, predicts its label
    # and returns it.
    
    def predict(self, x):
        min = math.inf
        prediction = math.inf
        for i in range(len(self.Xtrain)):
            dist = abs(self.E_len(x) - self.E_len(self.Xtrain[i]))
            if min> dist:
                min = dist
                prediction = self.ytrain[i]
        return prediction
    #____________________________________________________________________________________________________________________________
    
    # Here is predictions() method of the NearestNeighbors class.
    # It takes one argument, testset, which is the test data whose labels you want the model to predict.
    # It works by running the predict(x) method of the NearestNeighbors class in a for loop for each each 
    # element of Xtest, creates predictions for each element in Xtest, saves them in the ypred array and 
    # returns the ypred array of the NearestNeighbors class.
    
    def predictions(self, testset):
        self.ypred = np.empty(len(testset))
        for i in range(len(testset)):
            self.ypred[i] = self.predict(testset[i])
        return self.ypred
    
    #____________________________________________________________________________________________________________________________

    # Here is error_rate() method of the NearestNeighbors class.
    # It takes two arguments: testset, which is the test data whose labels you want the model to predict, 
    # and true_testset_labels, which are the true labels of the testdata.  

    def error_rate(self, testset, true_testset_lables):
        predicted_labels = self.predictions(testset)
        return 1 - accuracy_score(true_testset_lables, predicted_labels)



def main():
    NN = NearestNeighbors(X_iris_train, y_iris_train, X_iris_test, y_iris_test)
    print(type(NN.Xtrain[0]))
    print(NN.error_rate(NN.Xtrain, NN.ytrain))
    print(NN.error_rate(NN.Xtest, NN.ytest))
    
if __name__ == "__main__":
    main()