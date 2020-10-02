import random

import dimod
import optneal as opn
from openjij import SASampler


def main():
    """ Example of K-hot problem """
    N = 12
    K = 10
    numbers = [random.uniform(0, 5) for _ in range(N)]
    print(sorted(numbers))

    cost_dict = {i: numbers[i] for i in range(N)}
    cost = opn.Cost(cost_dict, shape=N)

    constraints = [({i: 1 for i in range(N)}, K)]
    penalty = opn.Penalty(constraints, shape=N)

    lam = 5.0
    cost_func = cost + lam * penalty.normalize()
    bqm = cost_func.to_dimod_bqm()

    sa_sampler = SASampler()
    lagrex_sampler = opt.LagrangeRelaxSampler(sa_sampler)
    sampleset = lagrex_sampler.sample(bqm, num_reads=10)

    # solver = dimod.ExactSolver()
    # results = solver.sample(bqm)

    # for sample in results.lowest().samples():
        # print([numbers[k] for k, v in sample.items() if v == 1])


if __name__ == '__main__':
    main()
