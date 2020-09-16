import random

import dimod
import genqubo as gq


def main():
    """ Example of Number Partiton problem """
    numbers =[2, 10, 3, 8, 5, 7, 9, 5, 3, 2]
    N = len(numbers)

    constraints = [({i: numbers[i] for i in range(N)}, 0)]
    F, C = gq.const_to_coeff(constraints, dims=N)
    h_mat, J_mat, offset = gq.convert_to_penalty(F, C, var_type=gq.SPIN)

    h_mat[0] += 1.0  # a bias to break symmetry
    bqm = gq.mat_to_dimod_bqm(h_mat, J_mat, offset, var_type=gq.SPIN)
    print(bqm)

    solver = dimod.ExactSolver()
    results = solver.sample(bqm)

    for i, sample in enumerate(results.lowest().samples()):
        group1 = [numbers[k] for k, v in sample.items() if v == 1]
        group2 = [numbers[k] for k, v in sample.items() if v == -1]
        print('sample {}:'.format(i), sum(group1), sum(group2), group1, group2)

if __name__ == '__main__':
    main()