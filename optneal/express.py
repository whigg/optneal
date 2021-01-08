import numbers

import numpy as np
import matplotlib.pyplot as plt
import dimod

from .core import dict_to_mat, const_to_coeff, convert_to_penalty


class Express:
    def __init__(self, mat, offset=0.0):
        self.mat = np.triu(mat) + np.tril(mat).T - np.diag(np.diag(mat))
        self.offset = offset

    @property
    def shape(self):
        return self.mat.shape

    def __add__(self, other):
        if isinstance(other, Express):
            if self.shape != other.shape:
                raise ValueError('incorrect shape of the expressions')

            concat_mat = self.mat + other.mat
            concat_offset = self.offset + other.offset

        elif isinstance(other, numbers.Real):
            concat_mat = self.mat.copy()
            concat_offset = self.offset + other
        else:
            raise ValueError('Express object can be added to Express object or real number')

        return Express(concat_mat, concat_offset)

    def __radd__(self, other):
        self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, numbers.Real):
            multp_mat = self.mat.copy()
            multp_mat *= other
            multp_offset = self.offset * other
        else:
            raise ValueError('Express object can be multiplied with real number')

        return Express(multp_mat, multp_offset)

    def __rmul__(self, other):
        return self.__mul__(other)

    def normalize(self, inplace=False):
        norm = np.max(np.abs(self.mat))
        _mat = self.mat / norm
        _offset = self.offset / norm

        if inplace:
            self.mat = _mat
        else:
            return Express(_mat, _offset)

    def to_dimod_bqm(self):
        Q_dict, offset = self.to_qubo()
        return dimod.BinaryQuadraticModel.from_qubo(Q_dict, offset)

    def to_qubo(self):
        Q_dict = {
            (int(i), int(j)): self.mat[i, j]
            for i, j in np.argwhere(self.mat != 0).astype(int)
        }
        return Q_dict, self.offset

    def show_qubo(self, save=False, fname='qubo.png'):
        plt.imshow(self.mat)
        plt.colorbar()
        plt.tight_layout()
        if save:
            plt.savefig(fname)
        plt.show()


class Cost(Express):
    def __init__(self, Q, shape):
        mat = dict_to_mat(Q, shape=shape)
        super(Cost, self).__init__(mat, 0)


class Penalty(Express):
    def __init__(self, constrs, shape):
        self.coeffs, self.consts = const_to_coeff(constrs, shape)
        mat, offset = convert_to_penalty(self.coeffs, self.consts)
        super(Penalty, self).__init__(mat, offset)


class Lagrange(Express):
    def __init__(self, constrs, multp, shape):
        F, C = const_to_coeff(constrs, shape)

        if isinstance(multp, np.ndarray):
            self.multp = multp
        elif isinstance(multp, list):
            self.multp = np.array(multp)
        elif isinstance(multp, numbers.Real):
            self.multp = np.ones(len(C)) * multp
        else:
            raise ValueError('multp must be numpy.ndarray, list or real number')

        mat = np.diag(np.dot(self.multp, F))
        offset = np.dot(self.multp, C)
        super(Lagrange, self).__init__(mat, offset)
