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
        elif isinstance(multi_index[0], int):
            return np.ravel_multi_index(multi_index, self.dims)
        else:
            return tuple([np.ravel_multi_index(mi, self.dims) for mi in multi_index])

    def unravel(self, index):
        """ Convert mono-index to multi-index """
        multi_index = np.unravel_index(index, self.dims)
        return multi_index[0] if len(multi_index) == 1 else multi_index


def check_input_mat(mat):
    if isinstance(mat, np.ndarray):
        return mat
    elif isinstance(mat, list):
        return np.array(mat)
    else:
        raise ValueError('The input matrix must be list or numpy.ndarray')

    
def convert_to_penalty(A, b):
    """ Convert an equation constraint to the the penalty form
    A * x = b -> || A * x - b ||_2^2 = x.T * Q * x + c,
    where Q = A.T * A - 2 * diag(b.T * A), c = b.T * b
    """
    
    _A = check_input_mat(A)
    _b = check_input_mat(b)

    if _A.shape[0] != _b.shape[0]:
        raise ValueError("can't dot A and b. ")

    Q = _A.T @ _A - 2 * np.diag((_b.T @ _A).reshape(-1))
    c = float(np.dot(_b.T, _b))
    return Q, c


def dict_to_mat(dict, dims):
    """ Convert a dict to a matrix """
    num_vars = np.prod(dims)
    mat = np.zeros((num_vars, num_vars))
    multi_idx = MultiIndex(dims)

    for k, v in dict.items():
        mat[multi_idx.ravel(k)] += v

    return mat


def const_to_coeff(constraints, dims):
    """ Convert constraints to coefficient matrix F and C
    F_{i, k} x_i ~ C_k for all k [~ represents ==, <=, or >=] -> F * x ~ C
    """
    num_cons = len(constraints)
    num_vars = np.prod(dims)
    multi_idx = MultiIndex(dims)

    F = np.zeros((num_cons, num_vars))
    C = np.array([c for _, c in constraints])
    for i, (f, _) in enumerate(constraints):
        for k, v in f.items():
            F[i, multi_idx.ravel(k)] += v

    return F, C


def mat_to_dimod_bqm(qubo_mat, offset):
    """ Convert QUBO matrix to Binary Quadratic Model of dimod library """

    def index_nonzero(matrix):
        """ Return an iterator on indices of non-zero components"""
        return zip(*np.argwhere(matrix).T.tolist())

    _qubo_mat = check_input_mat(qubo_mat)

    qubo_dict = {k: _qubo_mat[k] for k in index_nonzero(_qubo_mat)}
    return dimod.BinaryQuadraticModel.from_qubo(qubo_dict, offset)