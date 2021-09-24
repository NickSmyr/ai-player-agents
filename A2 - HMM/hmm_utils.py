import abc
import random
from typing import Tuple


class TNList(list, metaclass=abc.ABCMeta):
    def __init__(self, data: list):
        self.data = data
        list.__init__(self)

    def __getitem__(self, i):
        return self.data.__getitem__(i)

    def __setitem__(self, k, v):
        return self.data.__setitem__(k, v)

    def __repr__(self):
        return self.data.__repr__()

    def __str__(self):
        return self.data.__str__()

    def __len__(self):
        return self.data.__len__()

    def hadamard(self, l2: 'TNList') -> 'TNList':
        raise NotImplementedError


class Vector(TNList):
    """
    Assumed to be a column vector.
    """

    def __init__(self, data: list):
        assert type(data[0]) == float, f'Input not a float vector (type of data[0]={type(data[0])})'
        TNList.__init__(self, data=data)
        self.n = len(self.data)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.n, 1

    def normalize(self) -> None:
        s = sum(self.data)
        self.data = [self.data[i] / s for i in range(self.n)]

    def sum(self):
        return sum(self.data)

    def __matmul__(self, v2: 'Vector') -> float:
        """
        Perform dot product between self and v2.
        :param Vector v2: the second vector
        :return: a scalar result as a float
        """
        assert self.n == v2.n, 'Vector dims must be equal'
        return sum([self.data[i] * v2.data[i] for i in range(self.n)])

    def hadamard(self, v2: 'Vector' or list) -> 'Vector':
        result = [0.] * self.n
        v2_data = v2.data if type(v2) == Vector else v2
        for i in range(self.n):
            result[i] = self.data[i] * v2_data[i]
        return Vector(result)

    @staticmethod
    def random(n: int, normalize: bool = False) -> 'Vector':
        """
        Get a vector with elements drawn from a Uniform[0,1] distribution.
        :param int n: number of elements in vector
        :param bool normalize: set to True to normalize the vector to sum up to 1.0
        :return: a new Vector instance containing :attr:`n` random elements
        """
        v = Vector([random.random() for _ in range(n)])
        if normalize:
            v.normalize()
        return v


class Matrix2d(TNList):
    def __init__(self, data: list):
        assert len(data[1]) == len(data[0]), f'Dims not match len(data[0])={len(data[0])}, len(data[1])={len(data[1])}'
        TNList.__init__(self, data=data)
        self.data: list
        self.nrows = len(self.data)
        self.ncols = len(self.data[0])

    @property
    def shape(self) -> Tuple[int, int]:
        return self.nrows, self.ncols

    @property
    def T(self):
        return Matrix2d(list(zip(*self.data)))

    def sum_row(self, r: int):
        self.data: list
        return sum(self.data[r])

    def sum_rows(self):
        sums = [0.] * self.nrows
        for r in range(self.nrows):
            sums[r] = self.sum_row(r)
        return sums

    def normalize_rows(self) -> None:
        self.data: list
        for r in range(self.nrows):
            row_sum = sum(self.data[r])
            self.data[r] = [self.data[r][i] / row_sum for i in range(self.ncols)]

    def get_col(self, c: int) -> Vector:
        self.data: list
        vdata = [0.] * self.nrows
        for r in range(self.nrows):
            vdata[r] = self.data[r][c]
        return Vector(vdata)

    def get_row(self, r: int) -> Vector:
        self.data: list
        vdata = [0.] * self.ncols
        for c in range(self.ncols):
            vdata[c] = self.data[r][c]
        return Vector(vdata)

    def __matmul__(self, m2: 'Matrix2d' or Vector) -> 'Matrix2d' or Vector:
        """
        Perform dot product between self and m2.
        :param Matrix2d or Vector m2: the second matrix
        :return: a Matrix2d or Vector object as the result of matrix multiplication
        """
        self.data: list
        m2_rows, m2_cols = m2.shape
        assert self.ncols == m2_rows, f'Matrix dimensions must agree ({self.ncols} != {self.shape[0]})'

        # Initialize output list
        if type(m2) == Matrix2d:
            # Perform NAIVE matrix-matrix multiplication
            # m_result = [[0. for _ in range(m2_cols)] for _ in range(self.nrows)]
            # for i in range(self.nrows):
            #     for j in range(m2_cols):
            #         s = 0
            #         for k in range(self.ncols):
            #             s += self.data[i][k] * m2.data[k][j]
            #         m_result[i][j] = s
            # return Matrix2d(m_result)
            return Matrix2d([[sum(i * j for i, j in zip(r, c)) for c in zip(*m2.data)] for r in self.data])

        # Perform NAIVE matrix-vector multiplication
        # m_result = [0. for _ in range(m2_rows)]
        # for i in range(self.nrows):
        #     s = 0
        #     for j in range(self.ncols):
        #         s += self.data[i][j] * m2.data[j]
        #     m_result[i] = s
        return Vector([sum(i * j for i, j in zip(r, m2.data)) for r in self.data])

    def hadamard(self, m2: 'Matrix2d') -> 'Matrix2d':
        m_result = [[0. for _ in range(self.ncols)] for _ in range(self.nrows)]
        for i in range(self.nrows):
            for j in range(self.ncols):
                m_result[i][j] = self.data[i][j] * m2.data[i][j]
        return Matrix2d(m_result)

    @staticmethod
    def random(nrows: int, ncols: int, row_stochastic: bool = True) -> 'Matrix2d':
        """
        Initialize a 2d matrix with elements from uniform random in [0,1]
        :param int nrows: number of rows
        :param int ncols: number of columns
        :param bool row_stochastic: set to True to normalize each row of the matrix to sum up to 1.0
        :return: a 'Matrix2d' object
        """
        m = Matrix2d([[random.random() for _ in range(ncols)] for _ in range(nrows)])
        if row_stochastic:
            m.normalize_rows()
        return m

    # @staticmethod
    # def from_list(l: list, ncols: int) -> 'Matrix2d' or Vector:
    #     return Vector([l[li][0] for li in range(len(l))]) if ncols == 1 else Matrix2d(l)


if __name__ == '__main__':
    _v1 = Vector([1, 1, 1, 1])
    _v2 = Vector([0, 1, 0, 1])
    print(_v1 @ _v2)
