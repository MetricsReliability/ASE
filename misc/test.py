import numpy as np


def Rec(P, q):
    if q <= 0:
        return 1
    R = Rec(P, q / 2)
    R = R * R
    if np.mod(q, 2) == 0:
        return R
    else:
        return R * P


if __name__ == '__main__':
    print(Rec(5, 3))
