import numpy as np

class Perceptron:
    def __init__(self, num_features, learning_rate=0.01, max_epochs=100):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = np.zeros(num_features)
        self.bias = 0

    def train(self, X, y):
        for epoch in range(self.max_epochs):
            print("epoch " + str(epoch))
            for i in range(X.shape[0]):
                scores = np.dot(self.weights, X[i]) + self.bias
                predicted_class = 1 if scores >= 0 else 0
                true_class = y[i]

                if predicted_class != true_class:
                    update = self.learning_rate * (true_class - predicted_class)
                    self.weights += update * X[i]
                    self.bias += update

    def predict(self, X):
        scores = np.dot(X, self.weights) + self.bias
        return np.where(scores >= 0, 1, 0)