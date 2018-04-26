from perseptron import Perseptron

def learning():
    pers = Perseptron()
    examples = [[1, 1, 1],
                [-1, 1, 1],
                [1, -1, 1],
                [-1, -1, -1]]
    for i in range(0, 5):
        for example in examples:
            print("{0} {1} : {3}, correct {2}".format(*example, pers.get_output([example[0], example[1], 1])))
            pers.correct_weight(example[2])


learning()
