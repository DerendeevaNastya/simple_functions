from perseptron import Perseptron
import numpy as np
import copy

class Layer:
    def __init__(self, pers_count, activation, dendrites_count):
        self.layer = [Perseptron(activation, dendrites_count) for _ in range(pers_count)]
        self.size = pers_count
        self.last_input = []
        self.last_result = []

    def get_layer(self):
        return self.layer

    def get_deltas(self):
        return self.deltas

    def get_output(self, array):
        self.last_input = array
        result = [pers.get_output(array) for pers in self.layer]
        self.last_result = result
        return result

    def get_output_for_input_layer(self, array):
        self.last_input = array
        result = [self.layer[i].get_output([array[i]]) for i in range(len(self.layer))]
        self.last_result = result
        return result


    '''def get_weight_matrix(self):
        weigths = tuple([pers.weights for pers in self.layer])
        return np.vstack(weigths)'''

    def learn(self, delta_array):
        for i in range(self.size):
            self.layer[i].correct_weight(self.last_input, delta_array[i])

    def learn_input(self, delta_array):
        for i in range(self.size):
            self.layer[i].correct_weight([self.last_input[i]], delta_array[i])
