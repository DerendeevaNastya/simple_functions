from perseptron import Perseptron
from random import choice



examples = [[(1, 1), 0],
            [(0, 1), 1],
            [(1, 0), 1],
            [(0, 0), 0]]

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
        self.levels.append([Perseptron(like_sgn, 2), Perseptron(like_sgn, 2)])
        self.levels.append([Perseptron(like_sgn, 2)])


    def get_output(self, input_data):
        res1 = self.levels[0][0].get_output((input_data[0], 1 - input_data[1]))
        res2 = self.levels[0][1].get_output((1 - input_data[0], input_data[1]))
        self.levels[0][0].axon = res1
        self.levels[0][1].axon = res2
        return self.levels[1][0].get_output((res1, res2))

    def learning(self, X, y, result):
        old_weight = self.levels[1][0].weights
        tmpX = (self.levels[0][0].axon, self.levels[0][1].axon)
        self.levels[1][0].correct_weight(tmpX, y, result)
        new_weight = self.levels[1][0].weights

        self.levels[0][0].correct_weight(X, sgn(new_weight[0] - old_weight[0]), 0)
        self.levels[0][1].correct_weight(X, sgn(new_weight[1] - old_weight[1]), 0)

def test(network):
    for X, y in examples:
        result = network.get_output(X)
        print("{0} {1} : {3}, correct {2}".format(*X, y, result))

def learn(network):
    for _ in range(100):
        X, y = choice(examples)
        result = network.get_output(X)
        network.learning(X, y, result)




if __name__ == "__main__":
    xor = XorNetwork()
    
    learn(xor)
    test(xor)

