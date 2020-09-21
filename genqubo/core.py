import numpy as np
import dimod


BINARY = 'BINARY'
SPIN = 'SPIN'


class MultiIndex:
    """ Class for Multi-index """

    def __init__(self, dims):
        if isinstance(dims, int):
            self.dims = 1
            self.ravel = lambda x: x
        else:
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


def convert_to_penalty(A, b, var_type=BINARY):
    """ Convert an equation constraint to the the penalty form
    A * x = b -> || A * x - b ||_2^2 = x.T * Q * x + c,
    where Q = A.T * A - 2 * diag(b.T * A), c = b.T * b
    """

    _A = check_input_mat(A)
    _b = check_input_mat(b)

    if _A.shape[0] != _b.shape[0]:
        raise ValueError("can't dot A and b. ")

    c = float(np.dot(_b.T, _b))
    if var_type == BINARY:
        Q = _A.T @ _A - 2 * np.diag((_b.T @ _A).reshape(-1))
        return Q, c
    elif var_type == SPIN:
        h = - 2 * (_b.T @ _A).reshape(-1)
        J = _A.T @ _A
        c += np.sum(np.diag(J))
        J -= np.diag(np.diag(J))
        return h, J, c
    else:
        raise ValueError("var_type must be 'BINARY' or 'SPIN'")


def dict_to_mat(dict, dims):
    """ Convert a dict to a matrix """
    num_vars = np.prod(dims)
    mat = np.zeros((num_vars, num_vars))
    multi_idx = MultiIndex(dims)

    for k, v in dict.items():
        multi_k = multi_idx.ravel(k)
        if isinstance(multi_k, tuple):
            mat[multi_k] += v
        else:
            mat[multi_k, multi_k] += v

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
            multi_k = multi_idx.ravel(k)
            if isinstance(multi_k, tuple):
                if multi_k[0] != multi_k[1]:
                    raise ValueError('constraints must be linear')
                else:
                    F[i, multi_k[0]] += v
            else:
                F[i, multi_k] += v

    return F, C


def mat_to_dict(mat):
    """ Return an iterator on indices of non-zero components"""
    def f(x): return x[0] if len(x) == 1 else x
    return {f(k): mat[k] for k in zip(*np.argwhere(mat).T.tolist())}


def mat_to_dimod_bqm(h_mat=None, J_mat=None, Q_mat=None, offset=0.0, var_type=BINARY):
    """ Convert QUBO matrix to Binary Quadratic Model of dimod library """
    if var_type == BINARY:
        _Q_mat = check_input_mat(Q_mat)
        Q_dict = mat_to_dict(_Q_mat)
        return dimod.BinaryQuadraticModel.from_qubo(Q_dict, offset)
    elif var_type == SPIN:
        _h_mat = check_input_mat(h_mat)
        _J_mat = check_input_mat(J_mat)
        h_dict = mat_to_dict(_h_mat)
        J_dict = mat_to_dict(_J_mat)
        return dimod.BinaryQuadraticModel.from_ising(h_dict, J_dict, offset)
    else:
        raise ValueError("var_type must be 'BINARY' or 'SPIN'")
