import abc
import math
import random
import sys
from typing import Tuple, Optional, List

from constants import *
from player_controller_hmm import PlayerControllerHMMAbstract

N_HIDDEN = 1
WARMUP_STEPS = 75
N_MODELS = N_SPECIES
P_THRESHOLD = 1 / N_SPECIES
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


class HMM:
    """
    HMM Class:
    Our implementation of a 1-st order Hidden Markov Model.
    """

    def __init__(self, N: int, K: int, A: Optional[Matrix2d] = None, B: Optional[Matrix2d] = None,
                 pi: Optional[Vector] = None):
        """
        HMM class constructor.
        :param int N: number of hidden states
        :param int K: number of emission types
        :param (optional) A: the transmission model matrix
        :param (optional) B: the observation model matrix
        :param (optional) pi: the initial states pfm
        """
        # Random intialization
        self.A = A if A is not None else Matrix2d.random(N, N, row_stochastic=True)
        self.A_transposed = self.A.T
        self.B = B if B is not None else Matrix2d.random(N, K, row_stochastic=True)
        self.B_transposed = self.B.T
        self.pi = pi if pi is not None else Vector.random(N, normalize=True) if pi is None else pi
        self.N = N
        self.K = K

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
                          pi: Optional[Vector] = None,
                          T: Optional[int] = None) -> Tuple[float, List[Vector], List[float]]:
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
        # if T == 1:
        #     return 0., [], []
        # Initialize alpha
        alpha = pi.hadamard(B_T[observations[0]])
        # if alpha_tm1_sum == 0.0:
        #     raise RuntimeError(f'alpha_tm1 drove to 0.0 (t=0/{T})')
        c = 1 / (alpha.sum() + eps)
        # Store alphas (and Cs) in memory
        cs = [c, ]
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
            # alpha_tm1 = alpha
        # Return likelihood (sum of last alpha vec) and the recorded alphas
        return -Vector(cs).log_sum(), alphas, cs

    def beta_pass_scaled(self, observations: list, cs: Optional[List] = None, A: Optional[Matrix2d] = None,
                         B_T: Optional[Matrix2d] = None, T: Optional[int] = None) -> Tuple[float, List[Vector]]:
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
        if cs is None:
            cs = [1.] * T
        beta_tp1 = Vector([cs[-1]] * self.N)
        betas = [beta_tp1, ]
        # Iterate through reversed time
        for t in range(T - 2, -1, -1):
            #   - compute beta_t
            beta_t = A @ B_T[observations[t + 1]]
            beta_t = beta_t.hadamard(beta_tp1)
            #   - scale beta_t[i]
            beta_t *= cs[t]
            #   - append to betas list
            betas.append(beta_t)
            #   - save for next iteration
            beta_tp1 = beta_t
        # Betas are ordered in reverse to match scientific notations (betas[t] is really beta_t)
        betas.reverse()
        # Return likelihood, betas
        return 0., betas

    def gamma_pass(self, observations: list, alphas: Optional[List[Vector]] = None, cs: Optional[list] = None,
                   A: Optional[Matrix2d] = None, B_T: Optional[Matrix2d] = None,
                   T: Optional[int] = None) -> Tuple[List[Vector], List[Matrix2d]]:
        """
        Implementation of Baum-Welch algorithm's Gamma Pass to compute gammas & digammas (i.e. prob of being in state i
        at time t and moving to state j at time t+1).
        :param list observations: {Ot} for t=1...T, where Ot in {0, ..., K}
        :param (optional) alphas: output from alpha_pass
        :param (optional) cs: output from alpha_pass
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
        # if T == 1:
        #     return [], []
        # 1. Get alpha_t and beta_t for all t=0,...,T-1
        if alphas is None:
            _, alphas, cs = self.alpha_pass_scaled(observations, T=T)
        _, betas = self.beta_pass_scaled(observations, cs=cs, T=T)
        # 2. Compute digammas and gammas for every t
        gammas, digammas = [], []
        for t in range(T - 1):
            #   - compute digamma_t
            digamma = A.hadamard(alphas[t].outer(
                betas[t + 1].hadamard(B_T[observations[t + 1]])
            ))
            digammas.append(digamma)
            #   - marginalize over i (rows) to compute gamma_t
            gammas.append(digamma.sum_rows())
        # Add last gamma for time step T
        gammas.append(alphas[-1])
        return gammas, digammas

    def reestimate(self, observations: list, gammas: List[Vector], digammas: List[Matrix2d], lambda_mix: float = 1.0,
                   T: Optional[int] = None) -> Tuple[Optional[Vector], Optional[Matrix2d], Optional[Matrix2d]]:
        """
        Implementation of Baum-Welch algorithm's parameters re-estimation using computed gammas and digammas.
        Source: A Revealing Introduction to Hidden Markov Models, Mark Stamp
        :param list observations: {Ot} for t=1...T, where Ot in {0, ..., K}
        :param list gammas: computed from gamma_pass()
        :param list digammas: computed from gamma_pass()
        :param float lambda_mix: model averaging weight (A = newA * lambda_mix + A * (1.0-lambda_mix))
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
            # Compute (almost) common denominator
            denom = sum(gammas[t][i] for t in rT1)
            # Reestimate A
            A[i] = [0. for _ in rN] if denom == 0.0 else \
                [sum(digammas[t][i][j] for t in rT1) / denom for j in rN]
            # Reestimate B
            denom += gammas[T - 1][i]
            B[i] = [0. for _ in rK] if denom == 0.0 else \
                [sum(gammas[t][i] for t in range(T) if observations[t] == j) / denom for j in rK]

        # Normalize model parameters and return
        pi, A, B = Vector(pi), Matrix2d(A), Matrix2d(B)
        if lambda_mix == 1.0:
            return pi, A, B
        # Mix
        return pi * lambda_mix + self.pi * (1 - lambda_mix), A * lambda_mix + self.A * (1 - lambda_mix), \
               B * lambda_mix + self.B * (1 - lambda_mix)

    def baum_welch(self, observations: list, tol: float = 1e-3, max_iter: int = 30, T: Optional[int] = None,
                   update_params: bool = True, lambda_mix: float = 1.0) -> Optional[Tuple[Vector, Matrix2d, Matrix2d]]:
        """
        Implementation of Baum-Welch algorithm to estimate optimal HMM params (A, B) using successive gamma passes and
        re-estimation, until observations likelihood (alpha_pass) converges.
        :param list observations: {Ot} for t=1...T, where Ot in {0, ..., K}
        :param float tol: convergence criterion
        :param int max_iter: maximum allowed number of iterations
        :param (optional) T: total number of observations (in case list was pre-initialized but not fully filled)
        :param bool update_params:
        :param float lambda_mix:
        :return: a tuple object containing the (pi, A, B) matrices of the converged model
                 or None if update_params was set to True
        """
        if T is None:
            T = len(observations)
        A = self.A
        A_T = self.A_transposed
        B = self.B
        B_T = self.B_transposed
        pi = self.pi
        old_ll, ll, i = -math.inf, 0., 0
        while i < max_iter and ll > old_ll:
            # Forward pass -> alpha_t for t=0,...,T-1
            # alphas[t][i] ~ P(O_{0,...,t-1}, Xt=i | model)
            ll, alphas, cs = self.alpha_pass_scaled(observations, pi=pi, A_T=A_T, B_T=B_T, T=T)
            # Backward pass -> beta_t for t=0,...,T-1
            # betas[t][i] ~ P(O_{t,...,T-1} | Xt=i, model)
            _, beta = self.beta_pass_scaled(observations, cs=cs, A=A, B_T=B_T, T=T)
            # Combine to compute state probabilities -> gamma_t for t=0,...,T-1
            # digammas[t][i, j] ~ P(Xt=i, Xt+1=j | O_{t,...,T-1}, model)
            # gammas[t][i] ~ P(Xt=i | O_{t,...,T-1}, model)
            gammas, digammas = self.gamma_pass(observations, cs=cs, A=A, B_T=B_T, T=T)
            # Reestimate model parameters based on new state probabilities
            pi, A, B = self.reestimate(observations, gammas=gammas, digammas=digammas, lambda_mix=lambda_mix, T=T)
            A_T, B_T = A.T, B.T
            # Next Baum-Welch iteration (+ checks for convergence)
            i += 1
            if i == 1:
                continue
            elif i > 2 and abs(ll - old_ll) <= tol:
                break
            old_ll = ll
        # Update parameters and return
        if not update_params:
            return pi, A, B
        self.pi = pi
        self.A = A
        self.A_transposed = A.T
        self.B = B
        self.B_transposed = B.T


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
        self.models = None
        self.unexplored_fishes = None
        self.active_models = None
        self.obs_seq = None
        self.t = None
        super().__init__()

    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """
        self.models = [HMM(N=N_HIDDEN, K=N_EMISSIONS) for _ in range(N_MODELS)]
        self.unexplored_fishes = list(range(N_FISH))
        self.active_models = set()
        self.obs_seq = [[] for _ in range(N_FISH)]

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
        for fi in range(N_FISH):
            self.obs_seq[fi].append(observations[fi])
        # assert all(ot == self_ot for ot, self_ot in zip(observations, Matrix2d(self.obs_seq).get_col(self.t - 1)))
        # If we have enough data to start training, start the guessing procedure
        if step >= WARMUP_STEPS:
            #   - pick a fish from the unexplored ones randomly
            # fi = self.unexplored_fishes.pop(random.randint(0, len(self.unexplored_fishes) - 1))
            fi = self.unexplored_fishes.pop()
            #   - pass its observation sequence through all the models and select the most active one
            fi_probs = [model.alpha_pass_scaled(self.obs_seq[fi]) for model in self.models]
            # print(f'fi_probs={fi_probs}', file=stderr)
            return fi, argmax(fi_probs)[1]

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
        # Retrain model only if it made a wrong prediction
        if true_type not in self.active_models or not correct:
            self.active_models.add(true_type)
            self.models[true_type].baum_welch(self.obs_seq[fish_id], max_iter=30, tol=1e-8, update_params=True)
