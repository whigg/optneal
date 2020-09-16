import genqubo as gq


def main():
    """ Example of K-hot problem """
    N = 100
    K = 5
    constraints = [({i: 1 for i in range(N)}, K)]
    F, C = gq.const_to_coeff(constraints, dims=N)
    qubo_mat, offset = gq.convert_to_penalty(F, C)

    bqm = gq.mat_to_dimod_bqm(qubo_mat, offset)
    print(bqm)


if __name__ == '__main__':
    main()