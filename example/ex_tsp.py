import genqubo as gq


def main():
    """ Example of TSP """
    N = 10
    constraints = [({(i, j): 1 for j in range(N)}, 1) for i in range(N)]
    constraints += [({(i, j): 1 for i in range(N)}, 1) for j in range(N)]
    F, C = gq.const_to_coeff(constraints, dims=(N, N))
    qubo_mat, offset = gq.convert_to_penalty(F, C)

    bqm = gq.mat_to_dimod_bqm(qubo_mat, offset)
    print(bqm)


if __name__ == '__main__':
    main()