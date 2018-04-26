import random


class Perseptron:
    def __init__(self, activation, dendrites_count):
        self.weights = [random.random() for _ in range(dendrites_count + 1)]
        self.epsilon = 0.2
        self.activation = activation
        self.bias = 1

    def get_output(self, input_data):
        result = 0
        input_data = list(input_data) + [self.bias]
        for i in range(len(input_data)):
            result += input_data[i] * self.weights[i]
        result += self.weights[-1]
        return self.activation(result)

    def correct_weight(self, X, y, result):
        delta = y - result
        for i in range(len(X)):
            self.weights[i] += self.epsilon * delta * X[i]
        self.weights[-1] += self.epsilon * delta * self.bias