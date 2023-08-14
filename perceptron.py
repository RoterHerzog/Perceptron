import numpy as np


class Perceptron:

    def __init__(self, learning_rate = 0.01 , n_iterations = 100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.activation_func = self.unit_func


    def unit_func(self, x):
        return np.where(x>=0,1,0)
    
    def train(self,X,y):
        samples, features = X.shape
        self.bias = 0
        self.weights = np.zeros(features)
        y_ = np.array([1 if i >0 else 0 for i in y])

        for cap in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(X,self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)

                update = self.learning_rate* (y_[idx], y_predicted)
                self.weights += update * x_i
                self.bias += update 

    def predict(self,X):
        linear_output = np.dot(X,self.weights) # equals = wT (transpose of the weigts vector) 
        y_predicted = self.activation_function(linear_output)
        return y_predicted
    
