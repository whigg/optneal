import random

import dimod
import genqubo as gq


def main():
    """ Benchmark on the number of constraints """
    N = 1000
    K = 1000
    constraints = [({i: random.gauss(0, 1) for i in range(N)}, 1) for k in range(K)]
    F, C = gq.const_to_coeff(constraints, dims=N)
    qubo_mat, offset = gq.convert_to_penalty(F, C)

    # bqm = gq.mat_to_dimod_bqm(Q_mat=qubo_mat, offset=offset)
    # print(bqm)

if __name__ == '__main__':
    main()
