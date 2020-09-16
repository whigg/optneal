import unittest

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


if __name__ == '__main__':
    unittest.main()