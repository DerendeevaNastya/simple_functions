import math


def like_sgn(res, deriv=False):
    if deriv:
        return 1
    return 1 if res > 0.5 else 0


def S(x, deriv=False):
    if deriv:
        return S(x) * (1 - S(x))
    return 1 / (1 + math.exp(-x))
