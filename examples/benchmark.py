import random
import pickle

import dimod
import numpy as np
import genqubo as gq
from pyqubo import Array

from timer import execute_time


@execute_time
def make_model_genqubo(N):
    dims = (N, N)
    rmd = lambda: random.randint(1, 10)
    start_idx = 0
    cost_dict = {(start_idx, 0): -10}
    for i in range(N):
        for j in range(N):
            cost_dict.update({((i, N - 1), (j, 0)): rmd()})
            cost_dict.update({((i, t), (j, t + 1)): rmd() for t in range(N - 1)})

    cost_mat = gq.dict_to_mat(cost_dict, dims=dims)

    constraints = [({(i, t): 1 for t in range(N)}, 1) for i in range(N)]
    constraints += [({(i, t): 1 for i in range(N)}, 1) for t in range(N)]
    F, C = gq.const_to_coeff(constraints, dims=dims)
    cstr_mat, offset = gq.convert_to_penalty(F, C)

    lam = 5.0
    qubo_mat = cost_mat + lam * cstr_mat
    bqm = gq.mat_to_dimod_bqm(Q_mat=qubo_mat, offset=offset)


@execute_time
def make_model_pyqubo(N):
    bin_vars = Array.create(name='q', shape=(N, N), vartype='BINARY')
    rmd = lambda: random.randint(1, 10)
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
    bqm = model.to_dimod_bqm()


def main():
    """ Benchmark of TSP """

    prob_sizes = np.array([5, 7, 10, 20, 50, 70, 100])
    times_genqubo = {}
    times_pyqubo = {}
    num_iter = 10
    for n in prob_sizes:
        print('N:', n)
        times_genqubo[n] = []
        times_pyqubo[n] = []
        for _ in range(num_iter):
            times_genqubo[n].append(make_model_genqubo(n))
            times_pyqubo[n].append(make_model_pyqubo(n))

    with open('times_genqubo.pkl', 'wb') as f:
        pickle.dump(times_genqubo, f)
    with open('times_pyqubo.pkl', 'wb') as f:
        pickle.dump(times_pyqubo, f)


if __name__ == '__main__':
    main()