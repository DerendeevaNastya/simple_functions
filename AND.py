from perseptron import Perseptron
from random import choice

examples = [[(1, 1), 1],
            [(0, 1), 0],
            [(1, 0), 0],
            [(0, 0), 0]]


def like_sgn(res):
    return 1 if res >= 0.5 else 0


def learn(pers):
    for _ in range(100):
        X, y = choice(examples)
        result = pers.get_output(X)
        pers.correct_weight(X, y, result)

def test(pers):
    for X, y in examples:
        result = pers.get_output(X)
        print("{0} {1} : {3}, correct {2}".format(*X, y, result))
    print(pers.weights)


if __name__ == "__main__":
    pers = Perseptron(like_sgn, 2)

    learn(pers)
    test(pers)


