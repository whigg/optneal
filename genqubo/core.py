import numpy as np
import dimod


class MultiIndex:
    """ Class for Multi-index """
    def __init__(self, dims):
        self.dims = dims

    def ravel(self, multi_index):
        """ Convert multi-index to mono-index """
        if isinstance(multi_index, int):
            return multi_index
        else:
            return np.ravel_multi_index(multi_index, self.dims)

    def unravel(self, index):
        """ Convert mono-index to multi-index """
        return np.unravel_index(index, self.dims)


def convert_to_penalty(A, b):
    """ Convert an equation constraint to the the penalty form
    A * x = b -> || A * x - b ||_2^2 = x.T (A.T * A - 2 diag(b.T * A)) x + b.T * b
    """
    if A.shape[0] != b.shape[0]:
        raise ValueError("can't dot A and b. ")
    return A.T @ A - 2 * np.diag((b.T @ A).reshape(-1)), float(np.dot(b.T, b))


def const_to_coeff(constraints, dims):
    """ Convert constraints to coefficient matrix F and C
    F_{i, k} x_i ~ C_k for all k [~ represents ==, <=, or >=] -> F * x ~ C
    """
    num_cons = len(constraints)
    num_vars = np.prod(dims)
    multi_idx = MultiIndex(dims)

    F = np.zeros((num_cons, num_vars))
    C = np.array([c for _, c in constraints])
    for i, (f, c) in enumerate(constraints):
        for k, v in f.items():
            F[i, multi_idx.ravel(k)] += v

    return F, C


def mat_to_dimod_bqm(qubo_mat, offset):
    """ Convert QUBO matrix to Binary Quadratic Model of dimod library """

    def index_nonzero(matrix):
        """ Return an iterator on indices of non-zero components"""
        return zip(*np.argwhere(matrix).T.tolist())

    qubo_dict = {k: qubo_mat[k] for k in index_nonzero(qubo_mat)}
    return dimod.BinaryQuadraticModel.from_qubo(qubo_dict, offset)