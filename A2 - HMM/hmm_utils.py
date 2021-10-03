import abc
import math
import random
from typing import Tuple, List


class TNList(list, metaclass=abc.ABCMeta):
    def __init__(self, data: list):
        self.data = data
        list.__init__(self)

    @property
    @abc.abstractmethod
    def shape(self) -> Tuple[int, int]:
        raise NotImplementedError

    def __getitem__(self, i):
        return self.data.__getitem__(i)

    def __setitem__(self, k, v):
        return self.data.__setitem__(k, v)

    def __repr__(self):
        return self.data.__repr__()

    def __str__(self, round_places: int = -1, include_shape: bool = True):
        shape_str = self.shape.__str__() + ' ' if include_shape else ''
        if round_places == -1:
            return (shape_str + self.data.__str__()) \
                .replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace(',', '')
        data_str = ''
        for d in self.data:
            if type(d) == list:
                for dd in d:
                    data_str += str(round(dd, round_places)) + ' '
            else:
                data_str += str(round(d, round_places)) + ' '
        return (shape_str + data_str.rstrip()).replace('[', '').replace(']', '').replace('(', '').replace(')', '') \
            .replace(',', '').replace('\'', '')

    def __len__(self):
        return self.data.__len__()

    def __iter__(self):
        return iter(self.data)

    def append(self, o) -> None:
        self.data.append(o)

    def hadamard(self, l2: 'TNList') -> 'TNList':
        raise NotImplementedError

    def copy(self):
        return self.data.copy()


class Vector(TNList):
    """
    Vector Class:
    Our implementation of 1-d vectors. These are assumed to be COLUMN vectors.
    """

    def __init__(self, data: list, dtype=float):
        # Cast elements to float if not already casted
        if type(data[0]) == int and dtype == float:
            data = [float(d) for d in data]
        # Assert input is a 1-d list
        assert type(data[0]) == dtype, f'Input not a {dtype} vector (type(data[0])={type(data[0])} | dtype={dtype})'
        # Initialize wrapper
        TNList.__init__(self, data=data)
        self.n = len(self.data)

    @property
    def shape(self) -> Tuple[int, int]:
        return 1, self.n

    def dtype(self, dt) -> 'Vector':
        self.data = [dt(d) for d in self.data]
        return self

    def normalize(self) -> 'Vector':
        return self.__itruediv__(number=self.sum())

    def sum(self) -> float:
        return sum(self.data)

    def log_sum(self) -> float:
        return sum(map(math.log10, self.data))

    def __add__(self, v2: 'Vector') -> 'Vector':
        return Vector([v1d + v2d for v1d, v2d in zip(self.data, v2.data)])

    def __iadd__(self, v2: 'Vector') -> 'Vector':
        self.data = [v1d + v2d for v1d, v2d in zip(self.data, v2.data)]
        return self

    def __mul__(self, scalar: float) -> 'Vector':
        """
        Perform vector-scalar multiplication (scaling) and return self pointer.
        :param float scalar: the multiplier
        :return: self instance having first been scaled by the given scalar
        """
        return Vector([d * scalar for d in self.data])

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
        """
        Perform Hadamard (aka element-wise) product among the elements of the self and v2 vectors.
        :param Vector v2: second operand
        :return: a new Vector instance of the same size as self and v2 and with elements the products of the
                 corresponding elements of both vectors
        """
        return Vector([d * v2d for d, v2d in zip(self.data, v2.data if type(v2) == Vector else v2)])

    def outer(self, v2: 'Vector') -> 'Matrix2d':
        """
        Given vectors:
            a = (a1, a2, ..., an),
            b = (b1, b2, ..., bm)
        Returns a matrix of shape NxM
            A = [a1b1, a1b2, ..., a1bm,
                  .                 .
                  .        .        .
                  .                 .
                 anb1, anb2, ..., anbm]
        :param Vector v2: the second operand
        :return: a new Matrix2d instance of shape NxM
        """
        return Matrix2d([[v1i * v2j for v2j in v2.data] for v1i in self.data])

    def __imul__(self, number: float):
        """
        In place multiplication by a number (i.e. v *= number, where v is a Vector instance).
        :param float number: the multiplier
        :return: self (since the operation happens in place)
        """
        self.data = [c * number for c in self.data]
        return self

    def __itruediv__(self, number: float):
        """
        In place division by a number (i.e. v /= number, where v is a Vector instance).
        :param float number: the divisor
        :return: self (since the operation happens in place)
        """
        self.data = [c / number for c in self.data]
        return self

    def __truediv__(self, number: float):
        """
        In place division by a number (i.e. v /= number, where v is a Vector instance).
        :param float number: the divisor
        :return: self (since the operation happens in place)
        """
        return Vector([c / number for c in self.data])

    @staticmethod
    def _from_str_get_list(line: str) -> list:
        line_data = [x for x in line.rstrip().split(" ")]
        nrows = int(line_data.pop(0))
        assert nrows == 1, f'Vector should be row-vectors (nrows={nrows})'
        ncols = int(line_data.pop(0))
        assert ncols == len(line_data), f'Given numbers of elements do not match (ncols={ncols} | ' \
                                        f'len(line_data)={len(line_data)})'
        return [float(lj) for lj in line_data]

    @staticmethod
    def from_str(line: str) -> 'Vector':
        return Vector(Vector._from_str_get_list(line=line))

    @staticmethod
    def random(n: int, normalize: bool = False) -> 'Vector':
        """
        Get a vector with elements drawn from a Uniform[0,1] distribution.
        :param int n: number of elements in vector
        :param bool normalize: set to True to normalize the vector to sum up to 1.0
        :return: a new Vector instance containing :attr:`n` random elements
        """
        v = Vector([(1. / n) + (0.01 if i == 0 else 0.) + 0.001 * random.random() for i in range(n)])
        if normalize:
            v.normalize()
        return v


class DeltaVector(Vector):
    def __init__(self, data: List[Tuple[float, int]]):
        Vector.__init__(self, data=[t[0] for t in data])
        self.argmax_data = [t[1] for t in data]


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
        assert len(data) == 1 or len(data[1]) == len(data[0]), f'Dims not match len(data[0])={len(data[0])}, ' \
                                                               f'len(data[1])={len(data[1])}'
        TNList.__init__(self, data=data)
        self.data: list
        self.nrows = len(self.data)
        self.ncols = len(self.data[0])

    @property
    def shape(self) -> Tuple[int, int]:
        return self.nrows, self.ncols

    @property
    def T(self):
        if self.ncols == 1:
            return Matrix2d([[r[0] for r in self.data], ])
        elif self.nrows == 1:
            return Matrix2d([[c, ] for c in self.data[0]])
        return Matrix2d(list(zip(*self.data)))

    def sum_row(self, r: int):
        self.data: list
        return sum(self.data[r])

    def sum_rows(self) -> Vector:
        # sums = [0.] * self.nrows
        # for r in range(self.nrows):
        #     sums[r] = self.sum_row(r)
        return Vector([sum(r) for r in self.data])

    def normalize_rows(self) -> 'Matrix2d':
        self.data: list
        for r in range(self.nrows):
            row_sum = sum(self.data[r])
            self.data[r] = [self.data[r][i] / row_sum for i in range(self.ncols)]
        return self

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

    def __matmul__(self, m2: 'Matrix2d' or Vector or list) -> 'Matrix2d' or Vector:
        """
        Perform dot product between self and m2.
        :param Matrix2d or Vector m2: the second matrix
        :return: a Matrix2d or Vector object as the result of matrix multiplication
        """
        # Matrix-matrix multiplication
        if type(m2) == Matrix2d:
            assert self.ncols == m2.nrows, f'Matrix dimensions must agree ({self.ncols} != {m2.nrows})'
            return Matrix2d([[sum(ri * cj for ri, cj in zip(r, c)) for c in zip(*m2.data)] for r in self.data])
        # Matrix-vector multiplication
        # assert self.ncols == m2.n, f'Matrix dimensions must agree ({self.ncols} != {m2.n})'
        return Vector([sum(ri * rj for ri, rj in zip(r, m2.data if type(m2) == Vector else m2)) for r in self.data])

    def hadamard(self, m2: 'Matrix2d') -> 'Matrix2d':
        """
        Perform element-wise (aka Hadamard) matrix multiplication.
        :param Matrix2d m2: second operand as a Matrix2d instance.
        :return: a new Matrix2d instance of the same dims with the result of element-wise matrix multiplication.
        """
        return Matrix2d([[drc * m2drc for drc, m2drc in zip(dr, m2dr)] for dr, m2dr in zip(self.data, m2.data)])

    def __iadd__(self, n_or_m: float or 'Matrix2d'):
        """
        In place division by a number (i.e. m /= number, where m is a Matrix2d instance).
        :param float or Matrix2d n_or_m: the second term (either a number or an entire matrix
        :return: self (since the operation happens in place)
        """
        if type(n_or_m) == Matrix2d:
            self.data = [[c1 + c2 for c1, c2 in zip(r1, r2)] for r1, r2 in zip(self.data, n_or_m.data)]
        else:
            self.data = [[c + n_or_m for c in r] for r in self.data]
        return self

    def __add__(self, n_or_m: float or 'Matrix2d') -> 'Matrix2d':
        """
        In place division by a number (i.e. m /= number, where m is a Matrix2d instance).
        :param float or Matrix2d n_or_m: the second term (either a number or an entire matrix
        :return: self (since the operation happens in place)
        """
        if type(n_or_m) == Matrix2d:
            return Matrix2d([[c1 + c2 for c1, c2 in zip(r1, r2)] for r1, r2 in zip(self.data, n_or_m.data)])
        return Matrix2d([[c + n_or_m for c in r] for r in self.data])

    def __mul__(self, n: float) -> 'Matrix2d':
        """
        In place division by a number (i.e. m /= number, where m is a Matrix2d instance).
        :param float n: the second term (a float)
        :return: self (since the operation happens in place)
        """
        return Matrix2d([[c * n for c in r] for r in self.data])

    def __itruediv__(self, number: float):
        """
        In place division by a number (i.e. m /= number, where m is a Matrix2d instance).
        :param float number: the divisor
        :return: self (since the operation happens in place)
        """
        self.data = [[c / number for c in r] for r in self.data]
        return self

    def apply_func(self, f) -> 'Matrix2d':
        """
        Apply a function to each matrix element.
        """
        new_data = [[f(col) for col in row] for row in self.data]
        return Matrix2d(new_data)

    def is_close(self, m2: 'Matrix2d', tol: float = 1e-3) -> bool:
        """
        Check if self matrix is close to given m2.
        :param Matrix2d m2: the second matrix
        :param float tol: tolerance less than which elements are considered equal
        :return: a bool object
        """
        if self.shape != m2.shape:
            return False
        import numpy as np
        return np.allclose(np.array(self.data), np.array(m2.data), rtol=tol, atol=tol)

    @staticmethod
    def _from_str_get_list(line: str) -> list:
        line_data = [x for x in line.rstrip().split(" ")]
        nrows = int(line_data.pop(0))
        ncols = int(line_data.pop(0))
        assert nrows * ncols == len(line_data), f'Given numbers of elements do not match ' \
                                                f'((nrows,ncols)={(nrows, ncols)} | len(line_data)={len(line_data)})'
        return [[float(line_data[j + i * ncols]) for j in range(ncols)] for i in range(nrows)]

    @staticmethod
    def from_str(line: str) -> 'Matrix2d':
        return Matrix2d(Matrix2d._from_str_get_list(line=line))

    @staticmethod
    def random(nrows: int, ncols: int, row_stochastic: bool = True) -> 'Matrix2d':
        """
        Initialize a 2d matrix with elements from uniform random in [0,1]
        :param int nrows: number of rows
        :param int ncols: number of columns
        :param bool row_stochastic: set to True to normalize each row of the matrix to sum up to 1.0
        :return: a 'Matrix2d' object
        """
        # TODO: better initialization than this one
        m = Matrix2d([[
            1 * (1. / ncols)
            + 0.01 * (1. if ci == ri else 0.)
            + 0.001 * random.random()
            for ci in range(ncols)] for ri in range(nrows)])
        if row_stochastic:
            m.normalize_rows()
        return m


def argmax(l: list) -> Tuple[float, int]:
    """
    Find the maximum value and also return the argmax from a list of floats.
    :param list l: input list of floats
    :return: a tuple object containing the (max, argmax) as float and int respectively
    """
    return max(zip(l, range(len(l))))
