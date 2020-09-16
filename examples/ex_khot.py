import random

import dimod
import genqubo as gq


def main():
    """ Example of K-hot problem """
    N = 10
    K = 2
    cost_dict = {i: random.gauss(0, 1) for i in range(N)}
    cost_mat = gq.dict_to_mat(cost_dict, dims=N)

    constraints = [({i: 1 for i in range(N)}, K)]
    F, C = gq.const_to_coeff(constraints, dims=N)
    cstr_mat, offset = gq.convert_to_penalty(F, C)

    lam = 5.0
    qubo_mat = cost_mat + lam * cstr_mat
    bqm = gq.mat_to_dimod_bqm(Q_mat=qubo_mat, offset=offset)
    print(bqm)

    solver = dimod.ExactSolver()
    results = solver.sample(bqm)

    for sample in results.lowest().samples():
        print({k: v for k, v in sample.items() if v == 1})


if __name__ == '__main__':
    main()