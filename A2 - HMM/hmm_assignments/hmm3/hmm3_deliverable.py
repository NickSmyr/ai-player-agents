import fileinput
import math
from typing import Optional, Tuple, List


class TNList(list):
    def __init__(self, data: list):
        self.data = data
        list.__init__(self)

    @property
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

    def sum(self):
        return sum(self.data)

    def product(self):
        p = 1.
        for n in self.data:
            p *= n
        return p

    def __add__(self, v2: 'Vector') -> 'Vector':
        return Vector([v1d + v2d for v1d, v2d in zip(self.data, v2.data)])

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
        return Vector([d * v2d for d, v2d in zip(self.data, v2.data)])

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
        nrows = int(line_data.pop(0))
        assert nrows == 1, f'Vector should be row-vectors (nrows={nrows})'
        ncols = int(line_data.pop(0))
        assert ncols == len(line_data), f'Given numbers of elements do not match (ncols={ncols} | ' \
                                        f'len(line_data)={len(line_data)})'
        return Vector([float(lj) for lj in line_data])


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

    def __matmul__(self, m2: 'Matrix2d' or Vector) -> 'Matrix2d' or Vector:
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
        assert self.ncols == m2.n, f'Matrix dimensions must agree ({self.ncols} != {m2.n})'
        return Vector([sum(ri * rj for ri, rj in zip(r, m2.data)) for r in self.data])

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

    def __itruediv__(self, number: float):
        """
        In place division by a number (i.e. m /= number, where m is a Matrix2d instance).
        :param float number: the divisor
        :return: self (since the operation happens in place)
        """
        self.data = [[c / number for c in r] for r in self.data]
        return self

    @staticmethod
    def from_str(line: str):
        line_data = [x for x in line.rstrip().split(" ")]
        nrows = int(line_data.pop(0))
        ncols = int(line_data.pop(0))
        assert nrows * ncols == len(line_data), f'Given numbers of elements do not match ' \
                                                f'((nrows,ncols)={(nrows, ncols)} | len(line_data)={len(line_data)})'
        return Matrix2d([[float(line_data[j + i * ncols]) for j in range(ncols)] for i in range(nrows)])

    # @staticmethod
    # def from_list(l: list, ncols: int) -> 'Matrix2d' or Vector:
    #     return Vector([l[li][0] for li in range(len(l))]) if ncols == 1 else Matrix2d(l)
    def apply_func(self, f) -> 'Matrix2d':
        """
        Apply a function to each matrix element.
        """
        new_data = [[f(col) for col in row] for row in self.data]
        return Matrix2d(new_data)


def argmax(l: list) -> Tuple[float, int]:
    """
    Find the maximum value and also return the argmax from a list of floats.
    :param list l: input list of floats
    :return: a tuple object containing the (max, argmax) as float and int respectively
    """
    return max(zip(l, range(len(l))))


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
        self.A = A
        if A is not None:
            self.A_transposed = self.A.T
        self.B = B
        if B is not None:
            self.B_transposed = self.B.T
        self.pi = pi

        self.N = N
        self.K = K

    # TODO numerically stable methods
    # TODO keep alphas, betas in memory for gamma pass
    def alpha_pass(self, observations: list) -> Tuple[float, List[Vector], List[float]]:
        """
        Perform a forward pass to compute the likelihood of an observation sequence.
        :param list observations: {Ot} for t=1...T, where Ot in {0, ..., K}
        :return: a tuple object containing the (marginalized) probability that the HMM emitted the given observations,
                 a list of all the alphas computed during the recursive computation (as Vector objects) and
                 a list of all c's (aka the scaling coefficients)
        """
        # Initialize alpha
        alpha_tm1 = self.pi.hadamard(self.B.get_col(observations[0]))
        c = 1 / alpha_tm1.sum()
        # Store alphas (and Cs) in memory
        cs = Vector([c, ])
        #   - scale a_0
        alpha_tm1 *= c
        alphas = [alpha_tm1, ]
        T = len(observations)
        # Perform alpha-pass iterations
        for t in range(1, T):
            #   - compute alpha
            alpha = (self.A_transposed @ alpha_tm1).hadamard(self.B.get_col(observations[t]))
            #   - scale alpha
            c = 1 / alpha.sum()
            alpha *= c
            #   - append to list
            alphas.append(alpha)
            cs.append(c)
            if c == 0.0:
                raise RuntimeError(f'll drove to 0.0 (t={t})')
            alpha_tm1 = alpha
            # print(alpha, alpha.sum(), file=stderr)
        # Return likelihood (sum of last alpha vec) and the recorded alphas
        return -sum([math.log10(c) for c in cs]), alphas, cs

    def beta_pass(self, observations: list, cs: Optional[List] = None) -> Tuple[float, List[Vector]]:
        """
        Perform a backward pass as another way to compute the likelihood of an observation sequence.
        :param list observations: {Ot} for t=1...T, where Ot in {0, ..., K}
        :param (optional) cs: {Ct} for t=1...T, where Ct is the scaling coefficient for alpha_t
        :return: a tuple object containing the (marginalized) probability that the HMM emitted the given observations
                 and a list of all the betas computed during the recursive computation (as Vector objects)
        """
        T = len(observations)
        # Initial beta is beta_{T-1}
        if cs is None:
            cs = [1.] * T
        beta_tp1 = Vector([cs[-1]] * self.N)
        betas = [beta_tp1, ]

        for t in range(T - 2, -1, -1):
            # O_tp1 = observations[t+1]
            #   - compute beta_t
            beta_t = self.A @ self.B.get_col(observations[t + 1]).hadamard(beta_tp1)
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

    def gamma_pass(self, observations: list, alphas: Optional[List[Vector]] = None,
                   cs: Optional[list] = None) -> Tuple[List[Vector], List[Matrix2d]]:
        """
        Implementation of Baum-Welch algorithm's Gamma Pass to compute gammas & digammas (i.e. prob of being in state i
        at time t and moving to state j at time t+1).
        :param list observations: {Ot} for t=1...T, where Ot in {0, ..., K}
        :param (optional) alphas: output from alpha_pass
        :param (optional) cs: output from alpha_pass
        :return: a tuple containing ([Gamma_t = Vector(P(Xt=i|O_1:t,HMM) for all i's) for t = 1...T],
                                     [DiGamma_t = Matrix2d(P(Xt=i,Xt+1=j|O_1:t,HMM) for all (i, j)'s) for t = 1...T])
        """
        if alphas is None:
            _, alphas, cs = self.alpha_pass(observations)
        _, betas = self.beta_pass(observations, cs=cs)
        T = len(observations)

        gammas = []
        digammas = []
        # We need one digamma for every t
        for t in range(T - 1):
            digamma = self.A.hadamard(
                alphas[t].outer(
                    self.B.get_col(observations[t + 1]).hadamard(betas[t + 1])
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
                   lambda_mix: float = 1.1) -> None:
        """
        Implementation of Baum-Welch algorithm's parameters re-estimation using computed gammas and digammas.
        Source: A Revealing Introduction to Hidden Markov Models, Mark Stamp
        :param list observations: {Ot} for t=1...T, where Ot in {0, ..., K}
        :param list gammas: computed from gamma_pass()
        :param list digammas: computed from gamma_pass()
        :return: a tuple containing the new pi, A, B as Vector, Matrix2d, Matrix2d objects respectively
        """
        T = len(observations)
        # Reestimate pi
        new_pi = (1.0 - lambda_mix) * self.pi + lambda_mix * gammas[0]
        for i in range(self.N):
            denom = 0.
            for t in range(T - 1):
                denom += gammas[t][i]
            # Reestimate A
            for j in range(self.N):
                numer = 0.
                for t in range(T - 1):
                    numer += digammas[t][i][j]
                self.A[i][j] = (1.0 - lambda_mix) * self.A[i][j] + lambda_mix * (numer / denom)
            # Reestimate B
            denom += gammas[T - 1][i]
            for j in range(self.K):
                numer = 0.
                for t in range(T):
                    if observations[t] == j:
                        numer += gammas[t][i]
                self.B[i][j] = (1.0 - lambda_mix) * self.B[i][j] + lambda_mix * (numer / denom)
        # Re-initialize model
        self.pi = new_pi.normalize()
        self.A.normalize_rows()
        self.B.normalize_rows()
        self.A_transposed = self.A.T
        self.B_transposed = self.B.T

    def baum_welch(self, observations: list, tol: float = 1e-5,
                   max_iter: int = 100) -> Tuple[Matrix2d, Matrix2d, float]:
        """
        Implementation of Baum-Welch algorithm to estimate optimal HMM params (A, B) using successive gamma passes and
        re-estimation, until observations likelihood (alpha_pass) converges.
        :param list observations: {Ot} for t=1...T, where Ot in {0, ..., K}
        :param float tol: convergence criterion
        :param int max_iter: maximum allowed number of iterations
        :return: a tuple object containing the (A, B, final_likelihood) matrices of the converged model
        """
        # Get initial likelihood
        old_ll, alphas, cs = self.alpha_pass(observations=observations)
        # Successively improve it
        for i in range(max_iter):
            #   - compute gammas
            gammas, digammas = self.gamma_pass(observations=observations, alphas=alphas, cs=cs)
            #   - update model
            self.reestimate(observations=observations, gammas=gammas, digammas=digammas)
            #   - check convergence
            ll, alphas, cs = self.alpha_pass(observations=observations)
            assert ll >= old_ll, f'[baum_welch] ll={ll} < old_ll={old_ll} (i={i})'
            ll_diff = ll - old_ll
            if ll_diff < 0:
                break
            elif ll_diff < tol:
                break
            else:
                old_ll = ll
        return self.A, self.B, old_ll

    def delta_pass(self, observations: list) -> Tuple[Vector, float]:
        """
        Implementation of Viterbi algorithm to compute the most probable state sequence for the given observation
        sequence.
        [Theory] HMMs can be used to maximize the expected number of correct states.
        [  >>  ] DP can be used to maximize the entire sequence probability.
        :param list observations: {Ot} for t=1...T, where Ot in {0, ..., K}
        :return: a tuple containing the most probable state sequence in a Vector object and the probability of the most
                 probable path as float object
        """
        delta = self.pi.hadamard(self.B.get_col(observations[0]))
        # deltas = [delta, ]
        deltas_argmax = []
        for t in range(1, len(observations)):
            delta = DeltaVector([argmax(delta.hadamard(self.A.get_col(i)) * self.B[i][observations[t]])
                                 for i in range(self.N)])
            # deltas.append(delta)
            deltas_argmax.append(delta.argmax_data)
        # Calculate states path
        states_path_prob, last_state_index = argmax(delta)
        states_path = [last_state_index, ]
        for i in range(len(deltas_argmax) - 1, -1, -1):
            # states_path.append(deltas[i].argmax_data[states_path[-1]])
            states_path.append(deltas_argmax[i][states_path[-1]])
        states_path.reverse()
        return Vector(states_path, dtype=int), states_path_prob

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
        return HMM(N=N, K=K, A=A, B=B, pi=pi), obs


if __name__ == '__main__':
    #   - initialize HMM
    _hmm, _observations = HMM.from_input(fileinput.input())
    #   - output best model estimate
    A_opt, B_opt, _ = _hmm.baum_welch(_observations, tol=1e-10, max_iter=50)
    print(A_opt.__str__(round_places=6))
    print(B_opt.__str__(round_places=6))
