from for_XOR import Perseptron
from random import choice
import math


examples = [[(1, 1), 0],
            [(0, 1), 1],
            [(1, 0), 1],
            [(0, 0), 0]]


def S(x):
    return 1 / (1 + math.exp(-x))


def like_sgn(res):
    return 1 if res >= 0.5 else 0


def sgn(number):
    if number == 0:
        return 0
    if number > 0:
        return 1
    return -1

# (x or not y) and (not x or y)

#  nX - p1 \
#   Y /     \
#            p3 --- answer
#   X \     /
#  nY - p2 /
class XorNetwork:
    def __init__(self):
        self.levels = []
        self.levels.append([Perseptron(S, 2), Perseptron(S, 2)])
        self.levels.append([Perseptron(S, 2)])

    def get_output(self, input_data):
        res1 = like_sgn(self.levels[0][0].get_output(input_data))
        res2 = like_sgn(self.levels[0][1].get_output(input_data))
        self.levels[0][0].axon = res1
        self.levels[0][1].axon = res2
        return like_sgn(self.levels[1][0].get_output((res1, res2)))

    def learning(self, X, y, result):
        delta10 = y - result

        delta00 = delta10 * self.levels[1][0].weights[0]
        delta01 = delta10 * self.levels[1][0].weights[1]

        self.levels[0][0].correct_weight(X, delta00)
        self.levels[0][1].correct_weight(X, delta01)
        self.levels[1][0].correct_weight(
            (self.levels[0][0].axon, self.levels[0][1].axon),
            delta10)


def test(network):
    for X, y in examples:
        result = network.get_output(X)
        print("{0} {1} : {3}, correct {2}".format(*X, y, result))


def learn(network):
    for i in range(10000):
        X, y = choice(examples)
        result = network.get_output(X)
        network.learning(X, y, result)
        #print(network.levels[0][0].weights)
        #print(network.levels[0][1].weights)
    #print(network.levels[1][0].weights)


if __name__ == "__main__":
    xor = XorNetwork()
    
    learn(xor)
    test(xor)
