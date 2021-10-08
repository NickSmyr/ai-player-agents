import abc
import fileinput
import math
import random
import sys
from copy import deepcopy
from typing import Optional, Tuple, List

from constants import *
from player_controller_hmm import PlayerControllerHMMAbstract

N_HIDDEN_HC = 13
N_HIDDEN_LC = 3
WARMUP_STEPS = N_STEPS - N_FISH
NEXT_FISH_POLICY = 'random'
NEXT_FISH_POLICY_MAX = 20
eps = sys.float_info.epsilon


def argmax(l: list) -> Tuple[float, int]:
    """
    Find the maximum value and also return the argmax from a list of floats.
    :param list l: input list of floats
    :return: a tuple object containing the (max, argmax) as float and int respectively
    """
    return max(zip(l, range(len(l))))


class TNList(list):
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

    def copy(self) -> 'TNList':
        return self.__class__(deepcopy(self.data))


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
        # assert type(data[0]) == dtype, f'Input not a {dtype} vector (type(data[0])={type(data[0])} | dtype={dtype})'
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
        # assert self.n == v2.n, 'Vector dims must be equal'
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
    def from_str(line: str):
        line_data = [x for x in line.rstrip().split(" ")]
        return Vector([float(lj) for lj in line_data])

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
        # assert len(data) == 1 or len(data[1]) == len(data[0]), f'Dims not match len(data[0])={len(data[0])}, ' \
        #                                                        f'len(data[1])={len(data[1])}'
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
            # assert self.ncols == m2.nrows, f'Matrix dimensions must agree ({self.ncols} != {m2.nrows})'
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
    def from_str(line: str):
        line_data = [x for x in line.rstrip().split(" ")]
        nrows = int(line_data.pop(0))
        ncols = int(line_data.pop(0))
        # assert nrows * ncols == len(line_data), f'Given numbers of elements do not match ' \
        #                                         f'((nrows,ncols)={(nrows, ncols)} | len(line_data)={len(line_data)})'
        return Matrix2d([[float(line_data[j + i * ncols]) for j in range(ncols)] for i in range(nrows)])

    # noinspection PyUnusedLocal
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
            1. * (1. / ncols)
            # + 0.01 * (1. if ci == ri else 0.)
            + 0.001 * random.random()
            for ci in range(ncols)] for ri in range(nrows)])
        if row_stochastic:
            m.normalize_rows()
        return m


class HMM:
    """
    HMM Class:
    Our implementation of a 1-st order Hidden Markov Model.
    """

    def __init__(self, N: int, K: int, ):
        """
        HMM class constructor.
        :param int N: number of hidden states
        :param int K: number of emission types
        """
        self.N = N
        self.K = K
        self.label = None
        self.A = None
        self.A_transposed = None
        self.B = None
        self.B_transposed = None
        self.pi = None
        self.last_fish_id = None
        self.last_i = None
        self.last_ll = None

    def initialize_static(self, A: Matrix2d, B: Matrix2d, pi: Vector) -> None:
        """
        Initialize model matrices from given ones.
        :param Matrix2d A:
        :param Matrix2d B:
        :param Vector pi:
        """
        self.A = A.normalize_rows()
        self.A_transposed = self.A.T
        self.B = B.normalize_rows()
        self.B_transposed = self.B.T
        if pi is None:
            pi = Vector([1. / self.N] * self.N)
        self.pi = pi.normalize()

    def initialize(self, globalB: list, label: Optional[int] = None) -> None:
        """
        Initializes model matrices before training.
        :param list globalB: of shape (1,K)
        :param int label: e.g. the gt fish_type that this HMM instance is assumed to recognize
        """
        # Initialize B from observations
        B = Matrix2d([globalB.copy() for _ in range(self.N)])
        # Initialize A, pi
        A = Matrix2d.random(self.N, self.N, row_stochastic=True)
        pi = Vector.random(self.N, normalize=True)
        self.initialize_static(A=A, B=B, pi=pi)
        #   - set ground-truth label
        if label is not None:
            self.label = label

    def alpha_pass(self, observations: list, T: Optional[int] = None) -> float:
        """
        Naive forward pass (no Stamp's scaling)
        :param list observations: the observations sequence as a list object
        :param int T: the observations sequence's length as an integer
        :return: the probability of the observations given the model as a float
        """
        if T is None:
            T = len(observations)
        # if T == 1:
        #     return 0.0
        # Initialize alpha
        alpha = self.pi.hadamard(self.B_transposed[observations[0]])
        # if alpha_tm1.sum() == 0.0:
        #     # raise RuntimeError(f'alpha_tm1 drove to 0.0 (t=0/{T})')
        #     return 0.0
        # Perform alpha-pass iterations
        for Ot in observations[1:T]:
            #   - compute alpha
            alpha = self.A_transposed @ alpha
            alpha = alpha.hadamard(self.B_transposed[Ot])
            #   - check for underflow
            # if alpha.sum() == 0.0:
            #     raise RuntimeError(f'll drove to 0.0 (t={t}/{T})')
        # Return probability (sum of last alpha vec)
        return alpha.sum()

    def alpha_pass_scaled(self, observations: list, A_T: Optional[Matrix2d] = None, B_T: Optional[Matrix2d] = None,
                          pi: Optional[Vector] = None, T: Optional[int] = None) \
            -> Tuple[float, List[Vector], List[float]]:
        """
        Perform a forward pass to compute the likelihood of an observation sequence.
        :param list observations: {Ot} for t=1...T, where Ot in {0, ..., K}
        :param Vector or None pi: Initial Distribution Vector
        :param Matrix2d or None A_T: Transmission Matrix TRANSPOSED
        :param Matrix2d or None B_T: Emission Matrix TRANSPOSED
        :param (optional) T: total number of observations (in case list was pre-initialized but not fully filled)
        :return: a tuple object containing the (marginalized) probability that the HMM emitted the given observations,
                 a list of all the alphas computed during the recursive computation (as Vector objects) and
                 a list of all c's (aka the scaling coefficients)
        """
        if pi is None or A_T is None:
            pi = self.pi
            A_T = self.A_transposed
            B_T = self.B_transposed
        if T is None:
            T = len(observations)
        # Initialize alpha
        alpha = pi.hadamard(B_T[observations[0]])
        # if alpha_tm1_sum == 0.0:
        #     raise RuntimeError(f'alpha_tm1 drove to 0.0 (t=0/{T})')
        c = 1 / (alpha.sum() + eps)
        # Store alphas (and Cs) in memory
        cs = [c, ]
        cs_log_sum = math.log10(c)
        #   - scale a_0
        alpha *= c
        alphas = [alpha, ]
        # Perform alpha-pass iterations
        for t in range(1, T):
            #   - compute alpha
            alpha = A_T @ alpha
            alpha = alpha.hadamard(B_T[observations[t]])
            # alpha_sum = alpha.sum()
            #   - check for underflow
            # if alpha_sum == 0.0:
            #     raise RuntimeError(f'll drove to 0.0 (t={t}/{T})')
            #   - scale alpha
            c = 1. / (alpha.sum() + eps)
            alpha *= c
            #   - append to list
            alphas.append(alpha)
            cs.append(c)
            cs_log_sum += math.log10(c)
            # alpha_tm1 = alpha
        # Return likelihood (sum of last alpha vec) and the recorded alphas
        return -cs_log_sum, alphas, cs

    def beta_pass_scaled(self, observations: list, A: Optional[Matrix2d], B_T: Optional[Matrix2d],
                         cs: Optional[List] = None, T: Optional[int] = None) -> Tuple[float, List[Vector]]:
        """
        Perform a backward pass as another way to compute the likelihood of an observation sequence.
        :param list observations: {Ot} for t=1...T, where Ot in {0, ..., K}
        :param (optional) cs: {Ct} for t=1...T, where Ct is the scaling coefficient for alpha_t
        :param Matrix2d or None A: Transmission Matrix
        :param Matrix2d or None B_T: Emission Matrix TRANSPOSED
        :param (optional) T: total number of observations (in case list was pre-initialized but not fully filled)
        :return: a tuple object containing the (marginalized) probability that the HMM emitted the given observations
                 and a list of all the betas computed during the recursive computation (as Vector objects)
        """
        if A is None:
            A = self.A
            B_T = self.B_transposed
        if T is None:
            T = len(observations)
        # if T == 1:
        #     return 0., []
        # Initial beta is beta_{T-1}
        # if cs is None:
        #     cs = [1.] * T
        # beta_tp1 = Vector([cs[-1]] * self.N)
        # betas = [beta_tp1, ]
        # # Iterate through reversed time
        # for t in range(T - 2, -1, -1):
        #     #   - compute beta_t
        #     beta_t = A @ beta_tp1.hadamard(B_T[observations[t + 1]])
        #     #   - scale beta_t[i]
        #     beta_t *= cs[t]
        #     #   - append to betas list
        #     betas.append(beta_t)
        #     #   - save for next iteration
        #     beta_tp1 = beta_t
        #
        # # Betas are ordered in reverse to match scientific notations (betas[t] is really beta_t)
        # betas.reverse()
        # # Return likelihood, betas
        # return 0., betas
        if cs is None:
            cs = [1.] * T
        beta_tp1 = Vector([cs[-1]] * self.N)
        betas = [beta_tp1, ]

        for t in range(T - 2, -1, -1):
            # O_tp1 = observations[t+1]
            #   - compute beta_t
            beta_t = A @ beta_tp1.hadamard(B_T[observations[t + 1]])
            # beta_t = [0.] * self.N
            # for i in range(self.N):
            #     #   - compute beta_t[i]
            #     for j in range(self.N):
            #         beta_t[i] += self.A[i][j]*self.B[j][O_tp1]* beta_tp1[j]
            #   - scale beta_t[i]
            beta_t *= cs[t]
            #   - append to betas list
            betas.append(beta_t)
            #   - save for next iteration
            beta_tp1 = beta_t
        # for t in range(T - 2, -1, -1):
        #     beta = self.A @ self.B.get_col(observations[t + 1]).hadamard(beta)
        #     beta /= cs[t]
        #     betas.append(beta)
        # beta_{-1} Used only for testing purposes
        # betas.append(beta.hadamard(self.pi).hadamard(self.B.get_col(observations[0])))
        # Betas are ordered in reverse to match scientific notations (betas[t] is really beta_t)
        betas.reverse()
        # Return likelihood, betas
        return betas[0].sum(), betas

    def gamma_pass_scaled(self, observations: list, A: Optional[Matrix2d], B_T: Optional[Matrix2d],
                          alphas: List[Vector], betas: List[Vector],
                          T: Optional[int] = None) -> Tuple[List[Vector], List[Matrix2d]]:
        """
        Implementation of Baum-Welch algorithm's Gamma Pass to compute gammas & digammas (i.e. prob of being in state i
        at time t and moving to state j at time t+1).
        :param list observations: {Ot} for t=1...T, where Ot in {0, ..., K}
        :param (optional) alphas: output from alpha_pass
        :param (optional) betas: output from alpha_pass
        :param Matrix2d or None A: Transmission Matrix
        :param Matrix2d or None B_T: Emission Matrix TRANSPOSED
        :param (optional) T: total number of observations (in case list was pre-initialized but not fully filled)
        :return: a tuple containing ([Gamma_t = Vector(P(Xt=i|O_1:t,HMM) for all i's) for t = 1...T],
                                     [DiGamma_t = Matrix2d(P(Xt=i,Xt+1=j|O_1:t,HMM) for all (i, j)'s) for t = 1...T])
        """
        if A is None:
            A = self.A
            B_T = self.B_transposed
        if T is None:
            T = len(observations)
        # 1. Get alpha_t and beta_t for all t=0,...,T-1
        # if alphas is None:
        #     _, alphas, cs = self.alpha_pass_scaled(observations, T=T)
        # _, betas = self.beta_pass_scaled(observations, cs=cs, T=T)
        # 2. Compute digammas and gammas for every t
        # gammas = []
        # digammas = []
        # for t in range(T - 1):
        #     #   - compute digamma_t
        #     digamma = A.hadamard(alphas[t].outer(
        #         betas[t + 1].hadamard(B_T[observations[t + 1]])
        #     ))
        #     digammas.append(digamma)
        #     #   - marginalize over i (rows) to compute gamma_t
        #     gammas.append(digamma.sum_rows())
        # # Add last gamma for time step T
        # gammas.append(alphas[-1])
        # return gammas, digammas
        gammas = []
        digammas = []
        # We need one digamma for every t
        for t in range(T - 1):
            digamma = A.hadamard(
                alphas[t].outer(
                    betas[t + 1].hadamard(B_T[observations[t + 1]])
                )
            )
            # digamma /= ll
            digammas.append(digamma)
            gammas.append(digamma.sum_rows())

            # temp_matrix = alphas[t].outer(self.B.get_col(observations[t + 1]).hadamard(betas[t + 1]))
            # curr_digamma = self.A.hadamard(temp_matrix).apply_func(lambda x: x / ll)
            # digammas.append(curr_digamma)
            # gammas.append(curr_digamma.sum_rows())
            # assert all([[abs(c1 - c2) < 1e-10 for c1, c2 in zip(r1, r2)]
            #             for r1, r2 in zip(digamma.data, curr_digamma.data)])

        # Add last gamma for time step T
        gammas.append(alphas[-1])
        return gammas, digammas

    def reestimate(self, observations: list, gammas: List[Vector], digammas: List[Matrix2d],
                   T: Optional[int] = None, ) -> Tuple[Vector, Matrix2d, Matrix2d]:
        """
        Implementation of Baum-Welch algorithm's parameters re-estimation using computed gammas and digammas.
        Source: A Revealing Introduction to Hidden Markov Models, Mark Stamp
        :param list observations: {Ot} for t=1...T, where Ot in {0, ..., K}
        :param list gammas: computed from gamma_pass()
        :param list digammas: computed from gamma_pass()
        :param (optional) T: total number of observations (in case list was pre-initialized but not fully filled)
        :return: a tuple containing the new pi, A, B as Vector, Matrix2d, Matrix2d objects respectively
        """
        if T is None:
            T = len(observations)
        rN, rK, rT1 = range(self.N), range(self.K), range(T - 1)
        A = [[0. for _ in rN] for _ in rN]
        B = [[0. for _ in rK] for _ in rN]
        # Reestimate pi
        pi = gammas[0]
        for i in rN:
            # # Compute (almost) common denominator
            # denom = sum(gammas[t][i] for t in rT1)
            #
            # # Reestimate A
            # A[i] = [0. for _ in rN] if denom == 0.0 else \
            #     [sum(digammas[t][i][j] for t in rT1) / denom for j in rN]
            #
            # # Reestimate B
            # denom += gammas[T - 1][i]
            # B[i] = [0. for _ in rK] if denom == 0.0 else \
            #     [sum(gammas[t][i] for t in range(T) if observations[t] == j) / denom for j in rK]
            denom = 0.
            for t in range(T - 1):
                denom += gammas[t][i]
            # Reestimate A
            for j in range(self.N):
                numer = 0.
                for t in range(T - 1):
                    numer += digammas[t][i][j]
                A[i][j] = numer / denom
            # Reestimate B
            denom += gammas[T - 1][i]
            for j in range(self.K):
                numer = 0.
                for t in range(T):
                    if observations[t] == j:
                        numer += gammas[t][i]
                B[i][j] = numer / denom

        # Normalize model parameters and return
        return Vector(pi), Matrix2d(A), Matrix2d(B)

    def train(self, observations: list, max_iter: int = 100, p_tol: float = 1e-6, T: Optional[int] = None):
        """
        Implementation of Baum-Welch algorithm to estimate optimal HMM params (A, B) using successive gamma passes and
        re-estimation, until observations likelihood (alpha_pass) converges.
        :param list observations: {Ot} for t=1...T, where Ot in {0, ..., K}
        :param float p_tol: convergence criterion
        :param int max_iter: maximum allowed number of iterations
        :param (optional) T: total number of time steps
        :return: a tuple object containing the (A, B, final_likelihood) matrices of the converged model
        """
        A = self.A.copy()
        A_T = A.T
        B = self.B.copy()
        B_T = B.T
        pi = self.pi.copy()
        if T is None:
            T = len(observations)

        old_ll, ll, i = -math.inf, 0., 0
        for i in range(max_iter):
            ll, alphas, cs = self.alpha_pass_scaled(observations, A_T=A_T, B_T=B_T, pi=pi, T=T)
            _, betas = self.beta_pass_scaled(observations, A=A, B_T=B_T, cs=cs, T=T)
            #   - compute gammas
            gammas, digammas = self.gamma_pass_scaled(observations, A=A, B_T=B_T, alphas=alphas, betas=betas, T=T)
            #   - update model
            pi, A, B = self.reestimate(observations=observations, gammas=gammas, digammas=digammas)
            ll_diff = ll - old_ll
            if ll_diff < 0:
                print(f'[baum_welch][i={i:02d}] old_ll > ll Negative', file=sys.stderr)
                break
            elif ll_diff < math.exp(p_tol):
                break
            A_T = A.T
            B_T = B.T
            old_ll = ll

        # Update parameters and return
        # noinspection PyTypeChecker
        self.initialize_static(A=A.copy(), B=B.copy(), pi=pi.copy())
        self.last_i = i
        self.last_ll = ll
        return A, B, pi

    @staticmethod
    def from_input(finput: fileinput.FileInput) -> Tuple['HMM', Optional[List]]:
        """
        Initialize a new HMM instance using input provided in the form of Kattis text files.
        :param FileInput finput: a FileInput instance instantiated from stdin (e.g. by running python file using input
                                 redirect from a *.in file)
        :return: an HMM instance
        """
        A, B, pi, N, K, obs = None, None, None, None, None, None
        for i, line in enumerate(finput):
            if i == 0:
                A = Matrix2d.from_str(line)
            elif i == 1:
                B = Matrix2d.from_str(line)
                N, K = B.shape
            elif i == 2:
                pi = Vector.from_str(line)
            else:
                obs = Vector.from_str('1 ' + line).dtype(int).data
        hmm = HMM(N=N, K=K)
        hmm.initialize_static(A=A, B=B, pi=pi)
        return hmm, obs


class FishHMM:
    """
    FishHMM Class:
    Subclass of HMM to add functionality regarding fish updates and filtering.
    """

    def __init__(self, N_lc: int, N_hc: int, index: int):
        self.index = index
        self.active = False
        self.done = False
        self.n_found = 0
        self.lc_model = HMM(N=N_lc, K=N_EMISSIONS)
        self.hc_model = HMM(N=N_hc, K=N_EMISSIONS)

    def initialize(self, b, l=None):
        self.lc_model.initialize(b, l)
        self.hc_model.initialize(b, l)
        self.active = True

    def train(self, f: 'Fish', max_iter: int = 100, p_tol: float = 1e-6) -> None:
        # Initialize models
        self.initialize(f.beta)
        # Train low-capacity model
        self.lc_model.train(f.obs, max_iter=max_iter, p_tol=p_tol)
        # Train high-capacity model
        self.hc_model.train(f.obs, max_iter=max_iter, p_tol=p_tol)

    def infer(self, f: 'Fish') -> float:
        return max(self.lc_model.alpha_pass_scaled(observations=f.obs)[0],
                   self.hc_model.alpha_pass_scaled(observations=f.obs)[0])


class Fish:
    """
    Fish Class:
    Own implementation of a book-keeping struct to save statistics for each fish.
    """

    UNEXPLORED = 0
    EXPLORED = 1

    def __init__(self, index: int):
        """
        Fish class constructor.
        :param int index: fish index
        """
        self.index = index
        self._obs_seq = [-1] * N_STEPS
        self._t = 0
        self._state = Fish.UNEXPLORED
        self._species = None
        self._beta = [0] * N_EMISSIONS

    @property
    def obs(self) -> list:
        return self._obs_seq[:self._t]

    @obs.setter
    def obs(self, Ot: int) -> None:
        self._obs_seq[self._t] = Ot
        self._beta[Ot] += 1
        self._t += 1

    def get_most_probable(self, models: List[FishHMM], return_prob: bool = False):
        max_prob, max_mi = -math.inf, None
        for m in models:
            if m.active:
                prob = m.infer(self)
                if prob > max_prob:
                    max_prob = prob
                    max_mi = m.index
        # assert max_mi is not None
        if max_mi is None:
            max_mi = random.randint(0, N_SPECIES - 1)
        if return_prob:
            return max_prob, max_mi
        return max_mi

    @property
    def species(self) -> int:
        return self._species

    @species.setter
    def species(self, si: int) -> None:
        self._species = si
        self._state = Fish.EXPLORED

    @property
    def beta(self) -> list:
        return [1. / self._t for _ in range(N_EMISSIONS)]
        # return [float(b) / self._t + 0.1 * random.random() for b in self._beta]


class PlayerControllerHMM(PlayerControllerHMMAbstract):
    """
    PlayerControllerHMM Class:
    Our controller using 1 HMM / fish type, updating its parameters from given observations.
    -------------------------------
    FISH MOVES (POSSIBLE EMISSIONS)
    -------------------------------
      [4]   [0]  [5]
      [2]    F   [3]
      [6]   [1]  [7]
    -------------------------------
    """

    def __init__(self):
        self.fishes = None
        self.models = None
        self.unexplored_fis = None
        self.active_mis = None
        super().__init__()

    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """
        self.fishes = [Fish(index=fi) for fi in range(N_FISH)]
        self.models = [FishHMM(N_hc=N_HIDDEN_HC, N_lc=N_HIDDEN_LC, index=mi) for mi in range(N_SPECIES)]
        self.unexplored_fis = set(list(range(N_FISH)))
        self.active_mis = set()

    def pick_next_fish(self, policy: str = NEXT_FISH_POLICY) -> Tuple[int, Optional[int]]:
        """
        TODO
        :param policy:
        :return:
        """
        if policy == 'random':
            fo: Fish = self.fishes[random.choice(tuple(self.unexplored_fis))]
            return fo.index, fo.get_most_probable(self.models)
        if policy == 'sequential':
            fi = tuple(self.unexplored_fis)[0]
            fo: Fish = self.fishes[fi]
            return fo.index, fo.get_most_probable(self.models)
        if policy == 'max_all':
            # pick a fish from the unexplored ones STRATEGICALLY
            max_fi, max_fi_pred, max_fi_prob, counter = None, None, -math.inf, 0
            for fi in self.unexplored_fis:
                fo: Fish = self.fishes[fi]
                fi_prob, fi_mi = fo.get_most_probable(self.models, return_prob=True)
                if fi_prob > max_fi_prob:
                    max_fi = fi
                    max_fi_prob = fi_prob
                    max_fi_pred = fi_mi
                counter += 1
                if counter >= NEXT_FISH_POLICY_MAX:
                    break
            if max_fi is not None:
                # assert not math.isinf(max_fi_pred) and max_fi in self.unexplored_fis
                return max_fi, max_fi_pred
            return self.pick_next_fish('random')

    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """
        # Store observations
        for fi in self.unexplored_fis:
            fo: Fish = self.fishes[fi]
            fo.obs = observations[fi]
        # If we have enough data to start training, start the guessing procedure
        if step == WARMUP_STEPS:
            return random.randint(0, N_FISH - 1), random.randint(0, N_SPECIES - 1)
        # pick a fish from the unexplored ones using the pre-described policy
        elif step > WARMUP_STEPS:
            return self.pick_next_fish(policy=NEXT_FISH_POLICY)

    def reveal(self, correct, fish_id, true_type):
        """
        This methods gets called whenever a guess was made.
        It informs the player about the guess result
        and reveals the correct type of that fish.
        :param correct: tells if the guess was correct
        :param fish_id: fish's index
        :param true_type: the correct type of the fish
        :return:
        """
        fo: Fish = self.fishes[fish_id]
        model: FishHMM = self.models[true_type]
        if model.done:
            return

        # Update fish
        fo.species = true_type
        self.unexplored_fis.remove(fish_id)
        print(f'[reveal] fi={fish_id} | true_type={true_type} | correct={correct}', file=sys.stderr)

        # Check if model is done
        if model.n_found == (N_FISH / N_SPECIES - 1):
            self.active_mis.remove(true_type)
            model.done = True
            model.active = False
            return

        # Train model using the fish's observation sequence
        if (true_type not in self.active_mis and not self.models[true_type].done) or not correct:
            self.active_mis.add(true_type)
            model.n_found += 1
            model.train(f=fo, max_iter=40, p_tol=1e-6)
