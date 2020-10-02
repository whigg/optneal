import random
import pickle

import dimod
import numpy as np
import optneal as opn
from pyqubo import Array

from timer import execute_time


def rmd():
    return random.randint(1, 10)


@execute_time
def make_model_optneal(N):
    shape = (N, N)
    start_idx = 0
    cost_dict = {(start_idx, 0): -10}
    for i in range(N):
        for j in range(N):
            cost_dict.update({((i, N - 1), (j, 0)): rmd()})
            cost_dict.update({((i, t), (j, t + 1)): rmd() for t in range(N - 1)})
    cost = opn.Cost(cost_dict, shape)

    constraints = [({(i, t): 1 for t in range(N)}, 1) for i in range(N)]
    constraints += [({(i, t): 1 for i in range(N)}, 1) for t in range(N)]
    penalty = opn.Penalty(constraints, shape)

    lam = 5.0
    cost_func = cost + lam * penalty
    bqm = cost_func.to_dimod_bqm()


@execute_time
def make_model_pyqubo(N):
    bin_vars = Array.create(name='q', shape=(N, N), vartype='BINARY')
    start_idx = 0
    H_cost = -10 * bin_vars[start_idx, 0]
    for i in range(N):
        for j in range(N):
            H_cost += rmd() * bin_vars[i, N - 1] * bin_vars[j, 0]
            H_cost += np.sum([rmd() * bin_vars[i, t] * bin_vars[j, t + 1] for t in range(N - 1)])

    H_city = np.sum([(np.sum([bin_vars[i, t] for t in range(N)]) - 1)**2 for i in range(N)])
    H_time = np.sum([(np.sum([bin_vars[i, t] for i in range(N)]) - 1)**2 for t in range(N)])

    lam = 5.0
    H = H_cost + lam * (H_city + H_time)
    model = H.compile()
    bqm = model.to_bqm()


def main():
    """ Benchmark of TSP """

    prob_sizes = np.array([5, 7, 10, 20, 50, 70, 100])
    times_optneal = {}
    times_pyqubo = {}
    num_iter = 5
    for n in prob_sizes:
        print('N:', n)
        times_optneal[n] = []
        times_pyqubo[n] = []
        for _ in range(num_iter):
            times_optneal[n].append(make_model_optneal(n))
            times_pyqubo[n].append(make_model_pyqubo(n))

    with open('times_optneal.pkl', 'wb') as f:
        pickle.dump(times_optneal, f)
    with open('times_cpppyqubo.pkl', 'wb') as f:
        pickle.dump(times_pyqubo, f)


if __name__ == '__main__':
    main()
