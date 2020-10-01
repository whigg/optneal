import random

import numpy as np
import optneal as opn


def main():
    """ Example of Ohzeki Method """
    N = 10
    K = 2
    cost_dict = {i: random.uniform(0, 5) for i in range(N)}
    print('sum of exact sample:', sum(sorted(cost_dict.values())[:K]))
    cost = opn.Cost(cost_dict, shape=N)

    constraints = [({i: 1 for i in range(N)}, K)]
    lagrange = opn.Lagrange(constraints, multp=2.0, shape=N)
    print(lagrange.mat, lagrange.offset)

    lam = 5.0
    cost_func = cost + lam * lagrange
    bqm = cost_func.to_dimod_bqm()
    print(bqm)


if __name__ == '__main__':
    main()
