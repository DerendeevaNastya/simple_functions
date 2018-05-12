import random
import numpy as np



class Perseptron:
    def __init__(self, activation, dendrites_count):
        random.seed(3)
        self.weights = np.array([random.random() for _ in range(dendrites_count + 1)])
        self.epsilon = 0.2
        self.activation = activation
        self.bias = 1

    def get_output(self, input_data):
        input = np.array(list(input_data) + [self.bias])
        result = np.dot(input, self.weights)
        return self.activation(result)

    def correct_weight(self, X, delta):
        input = np.array(list(X) + [self.bias])
        deltas = input * self.epsilon * delta
        self.weights += deltas
