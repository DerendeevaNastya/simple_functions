import math

def like_sgn(res):
    return 1 if res >= 0.5 else 0


def S(x):
    return 1 / (1 + math.exp(-x))
