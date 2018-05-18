from perseptron import Perseptron
import numpy as np

class Layer:
    def __init__(self, pers_count, activation, dendrites_count):
        self.epsilon = 0.2
        self.activation = np.vectorize(activation)
        self.size = (pers_count, dendrites_count + 1)
        self.layer = np.random.random_sample(self.size)
        self.last_input = []
        self.last_result = []

    def get_layer(self):
        return self.layer

    def get_output(self, array):
        self.last_input = list(array) + [1]
        result = np.dot(self.layer, np.array(self.last_input))

        result = self.activation(result)

        self.last_result = result
        return result

    '''def get_output_for_input_layer(self, array):
        self.last_input = array
        result = [self.layer[i].get_output([array[i]]) for i in range(len(self.layer))]
        self.last_result = result
        return result'''


    '''def get_weight_matrix(self):
        weigths = tuple([pers.weights for pers in self.layer])
        return np.vstack(weigths)'''

    def learn(self, delta_array):
        inputs = np.array([self.last_input for _ in range(self.size[0])])
        deltas = (inputs.transpose() * self.epsilon * np.array(delta_array)).transpose()
        self.layer += deltas
        count = 0

    '''def learn_input(self, delta_array):
        for i in range(self.size):
            self.layer[i].correct_weight([self.last_input[i]], delta_array[i])
'''