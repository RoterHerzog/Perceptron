import perceptron
import numpy as np
import gzip
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from matplotlib import image as mpimg



with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

X_train, y_train = train_set[0], train_set[1]
X_test, y_test = test_set[0], test_set[1]


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled)

num_perceptrons = 10
perceptrons = []

for class_label in range(num_perceptrons):

    y_train_binary = np.where(y_train == class_label, 1, 0)
    y_test_binary = np.where(y_test == class_label, 1, 0)

    percept = perceptron.Perceptron(X_train_scaled.shape[1])
    percept.train(X_train_scaled, y_train_binary)
    perceptrons.append(percept)


predictions = np.array([percept.predict(X_test_scaled) for percept in perceptrons])
ensemble_predictions = np.argmax(predictions, axis=0)


ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
print(f"Ensemble Accuracy: {ensemble_accuracy:.2f}")

print(y_test)
print(ensemble_predictions)