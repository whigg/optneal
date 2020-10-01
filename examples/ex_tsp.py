import random

import dimod
import optneal as opn


def gen_random_cost(N, start_idx):
    def rmd():
        return random.randint(1, 10)
    cost_dict = {(start_idx, 0): -10}
    for i in range(N):
        for j in range(N):
            cost_dict.update({((i, N - 1), (j, 0)): rmd()})
            cost_dict.update({((i, t), (j, t + 1)): rmd() for t in range(N - 1)})

    return cost_dict


def main():
    """ Example of TSP """
    N = 4
    shape = (N, N)
    cost_dict = gen_random_cost(N, start_idx=0)
    cost = opn.Cost(cost_dict, shape)

    constraints = [({(i, t): 1 for t in range(N)}, 1) for i in range(N)]
    constraints += [({(i, t): 1 for i in range(N)}, 1) for t in range(N)]
    penalty = opn.Penalty(constraints, shape)

    lam = 10.0
    cost_func = cost + lam * penalty
    bqm = cost_func.to_dimod_bqm()

    solver = dimod.ExactSolver()
    results = solver.sample(bqm)

    mi = opn.MultiIndex(shape)
    for sample in results.lowest().samples():
        print({mi.unravel(k): v for k, v in sample.items() if v == 1})


if __name__ == '__main__':
    main()
