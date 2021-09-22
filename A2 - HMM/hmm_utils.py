import random
from typing import Tuple


class TNList(list):
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

    @staticmethod
    def random(n: int) -> 'Vector':
        return Vector([random.random() for _ in range(n)])


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

    @property
    def T(self):
        return Matrix2d(list(zip(*self.data)))

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
            m_result = [[0. for _ in range(m2_cols)] for _ in range(self.nrows)]
            # Perform NAIVE matrix-matrix multiplication
            for i in range(self.nrows):
                for j in range(m2_cols):
                    s = 0
                    for k in range(self.ncols):
                        s += self.data[i][k] * m2.data[k][j]
                    m_result[i][j] = s
            return Matrix2d(m_result)

        m_result = [0. for _ in range(m2_rows)]
        # Perform NAIVE matrix-vector multiplication
        for i in range(self.nrows):
            s = 0
            for j in range(self.ncols):
                s += self.data[i][j] * m2.data[j]
            m_result[i] = s
        return Vector(m_result)

    @staticmethod
    def random(nrows: int, ncols: int) -> 'Matrix2d':
        """
        Initialize a 2d matrix with elements from uniform random in [0,1]
        :param int nrows: number of rows
        :param int ncols: number of columns
        :return: a 'Matrix2d' object
        """
        return Matrix2d([[random.random() for _ in range(ncols)] for _ in range(nrows)])

    # @staticmethod
    # def from_list(l: list, ncols: int) -> 'Matrix2d' or Vector:
    #     return Vector([l[li][0] for li in range(len(l))]) if ncols == 1 else Matrix2d(l)


if __name__ == '__main__':
    _v1 = Vector([1, 1, 1, 1])
    _v2 = Vector([0, 1, 0, 1])
    print(_v1 @ _v2)
