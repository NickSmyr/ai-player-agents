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
        return self.data.__str__().replace('[', '').replace(']', '').replace(',', '')

    def __len__(self):
        return self.data.__len__()

    def __iter__(self):
        return iter(self.data)

    def hadamard(self, l2: 'TNList') -> 'TNList':
        raise NotImplementedError


class Vector(TNList):
    """
    Vector Class:
    Our implementation of 1-d vectors. These are assumed to be COLUMN vectors.
    """

    def __init__(self, data: list):
        # Cast elements to float if not already casted
        if type(data[0]) == int:
            data = [float(d) for d in data]
        # Assert input is a 1-d list
        assert type(data[0]) == float, f'Input not a float vector (type of data[0]={type(data[0])})'
        # Initialize wrapper
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

    def __mul__(self, scalar: float) -> 'Vector':
        """
        Perform vector-scalar multiplication (scaling) and return self pointer.
        :param float scalar: the multiplier
        :return: self instance having first been scaled by the given scalar
        """
        self.data = [d * scalar for d in self.data]
        return self

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __matmul__(self, v2: 'Vector') -> float:
        """
        Perform dot product between self and v2.
        :param Vector v2: the second vector
        :return: a scalar result as a float
        """
        assert self.n == v2.n, 'Vector dims must be equal'
        return sum([self.data[i] * v2.data[i] for i in range(self.n)])

    def hadamard(self, v2: 'Vector' or list) -> 'Vector':
        # result = [0.] * self.n
        # v2_data = v2.data if type(v2) == Vector else v2
        # for i in range(self.n):
        #     result[i] = self.data[i] * v2_data[i]
        return Vector([d * v2d for d, v2d in zip(self.data, v2.data)])

    def outer(self, v2: 'Vector') -> 'Matrix2d':
        """
        Given vectors:
            a = (a1, a2, ..., an),
            b = (b1, b2, ..., bm)
        Returns a matrix of shape NxM
            A = [ a1b1, a1b2, ..., a1bm,
                  .                  .
                  .         .        .
                  .                  .
                  anb1, anb2, ..., anbm]
        :param Vector v2: the second operand
        :return: a new Matrix2d instance of shape NxM
        """
        return Matrix2d([[v1i * v2j for v2j in v2.data] for v1i in self.data])

    def __str__(self):
        return f'1 {self.n} ' + super().__str__()

    @staticmethod
    def from_str(line: str):
        line_data = [x for x in line.rstrip().split(" ")]
        nrows = int(line_data.pop(0))
        assert nrows == 1
        n = int(line_data.pop(0))
        return Vector([float(line_data[j]) for j in range(n)])

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
    """
    Matrix2d Class:
    Our implementation of 2-d matrices.
    """

    def __init__(self, data: list):
        # Cast elements to float if not already casted
        if type(data[0][0]) == int:
            data = [[float(c) for c in r] for r in data]
        # Assert input is an orthogonal matrix
        assert len(data[1]) == len(data[0]), f'Dims not match len(data[0])={len(data[0])}, len(data[1])={len(data[1])}'
        # Initialize parent
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

    def sum_rows(self) -> Vector:
        sums = [0.] * self.nrows
        for r in range(self.nrows):
            sums[r] = self.sum_row(r)
        return Vector(sums)

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
            return Matrix2d([[sum(ri * cj for ri, cj in zip(r, c)) for c in zip(*m2.data)] for r in self.data])

        # Perform NAIVE matrix-vector multiplication
        return Vector([sum(ri * rj for ri, rj in zip(r, m2.data)) for r in self.data])

    def hadamard(self, m2: 'Matrix2d') -> 'Matrix2d':
        """
        Perform element-wise (aka Hadamard) matrix multiplication.
        :param Matrix2d m2: second operand as a Matrix2d instance.
        :return: a new Matrix2d instance of the same dims with the result of element-wise matrix multiplication.
        """
        return Matrix2d([[drc * m2drc for drc, m2drc in zip(dr, m2dr)] for dr, m2dr in zip(self.data, m2.data)])

    def __str__(self):
        return f'{self.nrows} {self.ncols} ' + super().__str__()

    @staticmethod
    def from_str(line: str):
        line_data = [x for x in line.rstrip().split(" ")]
        nrows = int(line_data.pop(0))
        ncols = int(line_data.pop(0))
        return Matrix2d([[float(line_data[j + i * ncols]) for j in range(ncols)] for i in range(nrows)])

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
    def apply_func(self, f) -> 'Matrix2d':
        """
        Apply a function to each matrix element.
        """
        new_data = [[f(col) for col in row] for row in self.data]
        return Matrix2d(new_data)


def outer_product(a: Vector, b: Vector) -> Matrix2d:
    """
    Given vectors a = (a1, a2, ..., an),
                  b = (b1, b2, ..., bm)
    Returns a matrix of shape NxM
                A = [ a1b1, a1b2, ..., a1bm,
                      .
                      .
                      .
                      anb1, anb2, ..., anbm]
    """
    n = a.shape[0]
    m = b.shape[0]

    data = [
        [a[i] * b[j] for j in range(m)]
        for i in range(n)
    ]
    return Matrix2d(data)


if __name__ == '__main__':
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
    print('PASSed')

    _line = '4 4 0.2 0.5 0.3 0.0 0.1 0.4 0.4 0.1 0.2 0.0 0.4 0.4 0.2 0.3 0.0 0.5'
    _m = Matrix2d.from_str(_line)
    assert _line == str(_m)
    print(_m)
