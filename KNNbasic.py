import numpy as np
import math
from sklearn.datasets import load_iris
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class KNearestNeighbors:

    # Here is the initalization method of the KNearestNeighbors class. It takes the 5 array arguments and saves
    # them as its own private variables, as shown below. 
    
    def __init__(self, Xtrain, ytrain, Xtest, ytest, k):
        self.Xtrain = Xtrain                          
        # Xtrain is the array that contains the data of the training set of the dataset.
        self.ytrain = ytrain                          
        # ytrain is the array that contains the true labels of the training set of the dataset.
        self.Xtest = Xtest                            
        # Xtest is the array that contains the data of the test set of the dataset.
        self.ytest = ytest                            
        # ytrain is the array that contains the true labels of the test set of the dataset.
        self.k = k
        # k is the number of neighbors this KNN algorithm will use in the prediction model.
        self.ypred = ""
        # ypred is an array that will be used to hold the predicted labels of each element of a given test set.

    #____________________________________________________________________________________________________________________________

    # Here is the E_len() method of the KNearestNeighbors class.
    # It takes a single argument, x, as an argument and calulates and returns its L2/Euclidean norm, i.e. distance from the origin.

    def E_len(self, x):
        return np.linalg.norm(x)
    
    #____________________________________________________________________________________________________________________________

    # Here is Number_of_labels() method of the KNearestNeighbors class.
    # It takes no arguments and by iterating through the ytrain array of the NearestNeighbors class, it creates
    # a list of all possible labels in the dataset, with no repeating labels.
    
    def Number_of_labels(self):
        labels = []
        for i in range(len(self.ytrain)):
            if self.ytrain[i] not in labels:
                labels.append(self.ytrain[i])
        return np.sort(np.array(labels))

    #____________________________________________________________________________________________________________________________

    # Here is the final_predict() method of the KNearestNeighbors class.
    # It takes two arguments, 'x' and 'k' and returns a single value which is the predicted label for the test data 'x'
    # based on the model. This method does this by first running the kpredict() method and saving the array it returns 
    # in a local variable called predictions. If k is 1, then the array contains only one value which is the final predition
    # automatically. If not, then if the array predtictions contains no repeating values, i.e. there is a tie, then the final_predict()
    # method is called again but 'k' is decremented. If there is no tie then the most common label in the array is chosen as
    # the final prediction and is returned.

    def final_predict(self, x, k):
        predictions = self.kpredict(x, k)
        l = self.Number_of_labels()
        if k == 1:
            return predictions[0]
        elif np.array_equal(predictions, np.unique(predictions)):
            return self.final_predict(x, k - 1)
        else:
            count = np.histogram(predictions, bins = l)
            return(np.argmax(count[0]))

    #____________________________________________________________________________________________________________________________

    # Here is the kpredict() method of the KNearestNeighbors class.
    # It takes a two arguments, 'x' and 'k' and returns an array of predictions numbering 'k'. It does this by copying
    # the training set and the set of its labels each to a local variable ,aug_trainset and aug_trainlabels. It will use
    # these arrays to then run the predict() method along with the parameter 'x' which will be a test datapoint whose
    # label we want to predict, 'k' number of times. Each time after a prediction is made, aug_trainset and aug_trainlabels
    # will be augmented by removing the values held at index 'index' which is returned along side the prediction from the
    # predict method.

    def kpredict(self, x, k):
        kprediction = np.empty(k)
        aug_trainset = self.Xtrain
        aug_trainlabels = self.ytrain
        for i in range(k):
            kprediction[i], index = self.predict(x, aug_trainset, aug_trainlabels)
            aug_trainset = np.delete(aug_trainset, index, 0)
            aug_trainlabels = np.delete(aug_trainlabels, index, 0)
        return kprediction

    #____________________________________________________________________________________________________________________________

    # Here is the predict() method of the KNearestNeighbors class.
    # It takes a three arguments, 'x', 'trainset' and 'trainlabels'. It goes through the trainset and finds 
    # the nearest neighbor to 'x' and sets that neighbors label, from the set 'trainlabels', to the prediction. 
    # It returns that label as the prediction along with the its index in the 'trainlabels' array.

    def predict(self, x, trainset, trainlabels):
        min = math.inf
        prediction = math.inf
        for i in range(len(trainset)):
            dist = abs(self.E_len(x) - self.E_len(trainset[i]))
            if min> dist:
                min = dist
                prediction = trainlabels[i]
                index = i
        return prediction, index

    #____________________________________________________________________________________________________________________________
    
    # Here is predictions() method of the KNearestNeighbors class.
    # It takes one argument, testset, which is the test data whose labels you want the model to predict.
    # It works by running the predict(x) method of the NearestNeighbors class in a for loop for each each 
    # element of Xtest, creates predictions for each element in Xtest, saves them in the ypred array and 
    # returns the ypred array of the NearestNeighbors class.
    
    def predictions(self, testset):
        self.ypred = np.empty(len(testset))
        for i in range(len(testset)):
            self.ypred[i] = self.final_predict(testset[i], self.k)
        return self.ypred
    
    #____________________________________________________________________________________________________________________________

    # Here is error_rate() method of the KNearestNeighbors class.
    # It takes two arguments: testset, which is the test data whose labels you want the model to predict, 
    # and true_testset_labels, which are the true labels of the testdata.  

    def error_rate(self, testset, true_testset_lables):
        predicted_labels = self.predictions(testset)
        return 1 - accuracy_score(true_testset_lables, predicted_labels)

#____________________________________________________________________________________________________________________________

# Here is cross_validation().
# It takes two arguments: Xtrain, which is the training data which will be split into the training and validation sets,
# and ytrain, which is the set of labels of the training data which will be split the same way as Xtrain.

def cross_validation(Xtrain, ytrain):
    min = math.inf
    best_fit = 0

    for i in range(1, 11):
        X_train, X_val, y_train, y_val = train_test_split(Xtrain, ytrain, test_size = 0.25, random_state = 3011)
        cvKNN = KNearestNeighbors(X_train, y_train, X_val, y_val, i)
        print("Error rate of cvKNN with k = ", i, ":")
        print(cvKNN.error_rate(cvKNN.Xtest, cvKNN.ytest), "\n")
        if min > cvKNN.error_rate(cvKNN.Xtest, cvKNN.ytest):
            min = cvKNN.error_rate(cvKNN.Xtest, cvKNN.ytest)
            best_fit = i
    return best_fit

#____________________________________________________________________________________________________________________________

# The following is the unfinished implementation of the k-folds cross-validation methods.
# It was planned to be down in two methods, kf_folds(), which would take 3 arguments: Xtrain, ytrain, folds
# and kf_cross_validation, which takes 4 arguments: X_val, y_val, X_train, y_train

# def kf_folds(Xtrain, ytrain, folds):
#     Xtrain_folds = np.array_split(Xtrain, folds)
#     ytrain_folds = np.array_split(ytrain, folds)
#     fold_k_err = np.empty(folds)
#     average_k_err = np.zeros(10)
#     for i in range(folds):
#         X_train = np.array([[]])
#         y_train = np.array([[]])
#         for j in range(folds):
#             if i != j:
#                 print(Xtrain_folds[j])
#                 print(X_train)
#                 X_train = np.concatenate((X_train, Xtrain_folds[j]), axis = 0)
#                 y_train = np.concatenate((y_train, ytrain_folds[j]), axis = 0)
#         X_val, y_val = Xtrain_folds[i], ytrain_folds[i]
#         fold_k_err[i] = kf_cross_validation(X_val, y_val, X_train, y_train)
#     for i in range(1, 11):
#         for j in range(folds):
#             average_k_err[i] += fold_k_err[j]
#         average_k_err = average_k_err/folds
#     return average_k_err


# def kf_cross_validation(X_val, y_val, X_train, y_train):
#     k_efficiencies = np.empty(10)
#     for i in range(1, 11):
#         KNN = KNearestNeighbors(X_train, y_train, X_val, y_val, i)
#         er = KNN.error_rate(KNN.Xtest, KNN.ytest)
#         print("|_____Error rate of KNN on with k = ", i, ":", "_____|")
#         print(er, "\n")
#         k_efficiencies[i-1] = er
#     return k_efficiencies

#____________________________________________________________________________________________________________________________

def main():
    iris = load_iris()
    X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(iris['data'], iris['target'], test_size=0.20, random_state = 3011)
    print(X_iris_train.shape, X_iris_test.shape, y_iris_train.shape, y_iris_test.shape)

    k = cross_validation(X_iris_train, y_iris_train)
    print("Best k for this dataset is: ", k)
    
if __name__ == "__main__":
    main()