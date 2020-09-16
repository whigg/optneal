import random

import dimod
import genqubo as gq


def main():
    """ Example of Number Partiton problem """
    numbers =[2, 10, 3, 8, 5, 7, 9, 5, 3, 2]
    N = len(numbers)

    constraints = [({i: numbers[i] for i in range(N)}, 0)]
    F, C = gq.const_to_coeff(constraints, dims=N)
    qubo_mat, offset = gq.convert_to_penalty(F, C, var_type=gq.SPIN)

    bqm = gq.mat_to_dimod_bqm(qubo_mat, offset)
    print(bqm)

    solver = dimod.ExactSolver()
    results = solver.sample(bqm)

    mi = gq.MultiIndex(dims=N)
    for sample in results.lowest().samples():
        print({mi.unravel(k): v for k, v in sample.items() if v == 1})


if __name__ == '__main__':
    main()