import unittest

import dimod
import genqubo as gq


class TestCore(unittest.TestCase):
    def test_core(self):
        N = 2
        constraints = [({(i, j): 1 for j in range(N)}, 1) for i in range(N)]
        constraints += [({(i, j): 1 for i in range(N)}, 1) for j in range(N)]
        F, C = gq.const_to_coeff(constraints, dims=(N, N))
        qubo_mat, offset = gq.convert_to_penalty(F, C)

        corr_qubo_mat = [[-2.0, 1.0, 1.0, 0.0],
                        [1.0, -2.0, 0.0, 1.0],
                        [1.0, 0.0, -2.0, 1.0],
                        [0.0, 1.0, 1.0, -2.0]]
        corr_offset = offset
        self.assertListEqual(qubo_mat.tolist(), corr_qubo_mat)
        self.assertEqual(offset, corr_offset)

    def test_multi_index(self):
        dims = (2, 2)
        multi_index = gq.MultiIndex(dims)
        idx = multi_index.ravel(1)
        self.assertEqual(idx, 1)

        idx = multi_index.ravel((1, 0))
        self.assertEqual(idx, 2)
        
        idx = multi_index.ravel(((0, 1), (1, 0)))
        self.assertEqual(idx, (1, 2))
        idx = multi_index.ravel(((1, 0), (1, 1)))
        self.assertEqual(idx, (2, 3))
        
    def test_convert_to_penalty(self):
        A = [[1, 1, 0, 0], [0, 0, 1, 1]]
        b = [1, 1]
        qubo_mat, offset = gq.convert_to_penalty(A, b)
        
        corr_qubo_mat = [[-1,  1,  0,  0],
                        [ 1, -1,  0,  0],
                        [ 0,  0, -1,  1],
                        [ 0,  0,  1, -1]]
        corr_offset = 2.0
        self.assertListEqual(qubo_mat.tolist(), corr_qubo_mat)
        self.assertEqual(offset, corr_offset)

    def test_dict_to_mat(self):
        N = 2
        qubo_dict = {((0, 0), (0, 1)): 1.0, ((1, 1), (1, 0)): 1.0, ((1, 0), (1, 0)): 1.0}
        qubo_mat = gq.dict_to_mat(qubo_dict, dims=(N, N))

        corr_qubo_mat = [[0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0]]
        self.assertListEqual(qubo_mat.tolist(), corr_qubo_mat)

    def test_const_to_coeff(self):
        N = 2
        constraints = [({(i, t): 1 for t in range(N)}, 1) for i in range(N)]
        constraints += [({(i, t): 1 for i in range(N)}, 1) for t in range(N)]
        F, C = gq.const_to_coeff(constraints, dims=(N, N))

        corr_F = [[1.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 1.0],
                 [1.0, 0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0, 1.0]]
        corr_C = [1.0, 1.0, 1.0, 1.0]
        self.assertListEqual(F.tolist(), corr_F)
        self.assertEqual(C.tolist(), corr_C)

    def test_mat_to_dimod_bqm(self):
        qubo_mat = [[-1,  1,  0,  0],
                    [ 1, -1,  0,  0],
                    [ 0,  0, -1,  1],
                    [ 0,  0,  1, -1]]
        offset = 2.0
        bqm = gq.mat_to_dimod_bqm(qubo_mat, offset=offset)
        corr_bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(qubo_mat, offset=offset)
        self.assertEqual(bqm, corr_bqm)


if __name__ == '__main__':
    unittest.main()