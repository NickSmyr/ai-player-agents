import unittest

from hmm_utils import Vector, Matrix2d


class TestHMMUtils(unittest.TestCase):
    def test_something(self):
        # _v1 = Vector([1, 1, 1, 1])
        # _v2 = Vector([0, 2, 0, 1])
        # print(_v1 @ _v2)
        # print(_v1.hadamard(_v2))
        # print(_v2 * 2)
        #
        # _m1 = Matrix2d([
        #     [1, 0, 0, 0],
        #     [0, 1, 0, 0],
        #     [0, 0, 1, 0],
        #     [0, 0, 0, 1],
        # ])
        # _m2 = Matrix2d([
        #     [1, 2, 0, 0],
        #     [0, 1, 2, 0],
        #     [1, -0.9, 1, 2],
        #     [0.01, 1, 0, 1],
        # ])
        # print(_m1.hadamard(_m2))
        # print(_m1 @ _m2)
        # print(_m2)

        _a = Vector([1., 2., 3.])
        # print(_a)
        _b = Vector([1., 10., 100.])
        # c = outer_product(a, b)
        _c = _a.outer(_b)
        _true = [[1, 10, 100],
                 [2, 20, 200],
                 [3, 30, 300]]
        for _i in range(3):
            for _j in range(3):
                assert _c[_i][_j] == _true[_i][_j]
        # print('PASSed')

        _line = '4 4 0.2 0.5 0.3 0.0 0.1 0.4 0.4 0.1 0.2 0.0 0.4 0.4 0.2 0.3 0.0 0.5'
        _m = Matrix2d.from_str(_line)
        assert _line == str(_m)
        # print(_m)



if __name__ == '__main__':
    unittest.main()
