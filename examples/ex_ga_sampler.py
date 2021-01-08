import numpy as np
import dimod

from optneal.ga_sampler import GASampler


def main():
    N = 50
    Q = {(i, j): np.random.normal() for i in range(N) for j in range(i, N)}
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    Q2 = {k: np.random.rand() * v for k, v in Q.items()}
    bqm2 = dimod.BinaryQuadraticModel.from_qubo(Q2)

    sampler = GASampler()
    responses = sampler.sample(bqm_list=[bqm, bqm2], num_pops=50, cxpb=0.5, mutpb=0.2, num_gens=100)
    print(responses)


if __name__ == '__main__':
    main()
