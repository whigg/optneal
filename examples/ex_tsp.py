import random

import dimod
import genqubo as gq


def gen_random_cost(N, start_idx):
    rmd = lambda: random.randint(1, 10)
    cost_dict = {(start_idx, 0): -10}
    for i in range(N):
        for j in range(N):
            cost_dict.update({((i, N - 1), (j, 0)): rmd()})
            cost_dict.update({((i, t), (j, t + 1)): rmd() for t in range(N - 1)})
    
    return cost_dict


def main():
    """ Example of TSP """
    N = 4
    dims = (N, N)
    cost_dict = gen_random_cost(N, start_idx=0)
    cost_mat = gq.dict_to_mat(cost_dict, dims=dims)

    constraints = [({(i, t): 1 for t in range(N)}, 1) for i in range(N)]
    constraints += [({(i, t): 1 for i in range(N)}, 1) for t in range(N)]
    F, C = gq.const_to_coeff(constraints, dims=dims)
    cstr_mat, offset = gq.convert_to_penalty(F, C)

    lam = 5.0
    qubo_mat = cost_mat + lam * cstr_mat
    bqm = gq.mat_to_dimod_bqm(Q_mat=qubo_mat, offset=offset)
    print(bqm)

    solver = dimod.ExactSolver()
    results = solver.sample(bqm)

    mi = gq.MultiIndex(dims=dims)
    for sample in results.lowest().samples():
#         print(sample)
        print({mi.unravel(k): v for k, v in sample.items() if v == 1})


if __name__ == '__main__':
    main()