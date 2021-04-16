import numpy as np


def meanSquareError(yTrue, yPred):
    return np.mean(yTrue - yPred)


class LinearRegression:
    def __init__(self, n_iters = 1000, learning_rate=0.01):
        self.numberOfIterators = n_iters
        self.learningRate = learning_rate
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        # initializing the value of parameters
        numberofSample, numberofFeature = X.shape
        self.weights = np.zeros(numberofFeature)
        self.bias = 0

        for i in range(self.numberOfIterators):
            # predicting y value
            yHat = np.dot(X, self.weights) + self.bias

            # calculations of gradients
            dw = 1 / (numberofSample) * np.dot(X.T, (yHat - y))
            db = 1 / (numberofSample) * np.sum(yHat - y)

            # updating weights and bias
            self.weights -= self.learningRate * dw
            self.bias -= self.learningRate * db


    def predict(self, X):
        ypredicted = np.dot(X, self.weights) + self.bias
        return ypredicted
