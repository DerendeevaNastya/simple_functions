import random
import numpy as np


class Perseptron:
    def __init__(self, activation, dendrites_count):
        self.weights = np.array([random.random() for _ in range(dendrites_count + 1)])
        self.epsilon = 0.1
        self.activation = activation
        self.bias = 1

    def get_output(self, input_data):
        input_data = np.array(list(input_data) + [self.bias])
        result = np.dot(input_data, self.weights)
        return self.activation(result)

    def correct_weight(self, X, delta):
        input_data = np.array(list(X) + [self.bias])
        derivatives = np.array([self.activation(x)*(1 - self.activation(x)) for x in input_data])
        self.weights += input_data * self.epsilon * delta * derivatives
