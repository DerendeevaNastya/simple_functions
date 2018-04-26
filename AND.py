from perseptron import Perseptron
import random


def like_sgn(res):
    return 1 if res >= 0.5 else 0


def learning():
    pers = Perseptron(like_sgn, 2)
    examples = [[1, 1, 1],
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 0]]
    for _ in range(0, 300):
        j = random.randint(0, 3)
        result = pers.get_output([examples[j][0], examples[j][1]])
        print("{0} {1} : {3}, correct {2}".format(*examples[j], result))
        #result = like_sgn(result)
        pers.correct_weight(examples[j][2], result)


learning()


