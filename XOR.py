from random import choice
import actions
from layer import Layer
import numpy as np


examples = [[(1, 1), 0],
            [(0, 1), 1],
            [(1, 0), 1],
            [(0, 0), 0]]


# (x or not y) and (not x or y)

#  nX - p1 \
#   Y /     \
#            p3 --- answer
#   X \     /
#  nY - p2 /
class XorNetwork:
    def __init__(self):
        self.input_layer = Layer(2, actions.S, 2)
        self.output_layer = Layer(1, actions.S, 2)

    def get_output(self, X):
        input_result = self.input_layer.get_output(X)
        output_result = self.output_layer.get_output(input_result)
        return output_result[0]

    def learning(self, y, result):
        error2 = - (result - y) * result * (1 - result)

        error1 = (self.output_layer.layer[0, :-1] *
                  error2 *
                  (1 - np.array(self.input_layer.last_result)) *
                  np.array(self.input_layer.last_result))

        self.output_layer.learn([error2])
        self.input_layer.learn(error1.tolist())



def test(network):
    for X, y in examples:
        result = network.get_output(X)
        print("{0} {1} : {3}, correct {2}".format(*X, y, result))


def learn(network):
    for _ in range(100000):
        X, y = choice(examples)
        result = network.get_output(X)
        network.learning(y, result)


if __name__ == "__main__":
    xor = XorNetwork()

    learn(xor)
    test(xor)
    #print(xor.input_layer.get_layer()[0].weights)
    #print(xor.input_layer.get_layer()[1].weights)
    #print(xor.output_layer.get_layer()[0].weights)
