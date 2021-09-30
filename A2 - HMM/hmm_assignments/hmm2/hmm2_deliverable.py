import fileinput
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

    def __str__(self, round_places: int = -1, include_shape: bool = False):
        shape_str = self.shape.__str__() + ' ' if include_shape else ''
        if round_places == -1:
            return (shape_str + self.data.__str__()) \
                .replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace(',', '')
        return (shape_str + [f'{d:.{round_places}f}' for d in self.data].__str__()) \
            .replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace(',', '').replace('\'', '')

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
    def alpha_pass(self, observations: list) -> Tuple[float, List[Vector]]:
        """
        Perform a forward pass to compute the likelihood of an observation sequence.
        :param list observations: {Ot} for t=1...T, where Ot in {0, ..., K}
        :return: a tuple object containing the (marginalized) probability that the HMM emitted the given observations
                 and a list of all the alphas computed during the recursive computation (as Vector objects)
        """
        # Initialize alpha
        alpha = self.pi.hadamard(self.B.get_col(observations[0]))
        # Store alphas in memory
        alphas = [alpha, ]
        # print(alpha, alpha.sum(), file=stderr)
        # Perform alpha-pass iterations
        for t in range(1, len(observations)):
            alpha = (self.A_transposed @ alpha).hadamard(self.B.get_col(observations[t]))
            alphas.append(alpha)
            # print(alpha, alpha.sum(), file=stderr)
        # Return likelihood (sum of last alpha vec) and the recorded alphas
        return alpha.sum(), alphas

    def beta_pass(self, observations: list) -> Tuple[float, List[Vector]]:
        """
        Perform a backward pass as another way to compute the likelihood of an observation sequence.
        :param list observations: {Ot} for t=1...T, where Ot in {0, ..., K}
        :return: a tuple object containing the (marginalized) probability that the HMM emitted the given observations
                 and a list of all the betas computed during the recursive computation (as Vector objects)
        """
        T = len(observations)
        # Initial beta is beta_{T-1}
        beta = Vector([1.] * self.N)
        betas = [beta, ]
        for t in range(T - 2, -1, -1):
            beta = self.A @ self.B.get_col(observations[t + 1]).hadamard(beta)
            betas.append(beta)
        # beta_{-1} Used only for testing purposes
        betas.append(beta.hadamard(self.pi).hadamard(self.B.get_col(observations[0])))
        # Betas are ordered in reverse to match scientific notations (betas[t] is really beta_t)
        betas.reverse()
        # Return likelihood, betas
        return betas[0].sum(), betas

    def gamma_pass(self, observations: list) -> Tuple[List[Vector], List[Matrix2d]]:
        """
        Implementation of Baum-Welch algorithm's Gamma Pass to compute gammas & digammas (i.e. prob of being in state i
        at time t and moving to state j at time t+1).
        :param list observations: {Ot} for t=1...T, where Ot in {0, ..., K}
        :return: a tuple containing ([Gamma_t = Vector(P(Xt=i|O_1:t,HMM) for all i's) for t = 1...T],
                                     [DiGamma_t = Matrix2d(P(Xt=i,Xt+1=j|O_1:t,HMM) for all (i, j)'s) for t = 1...T])
        """
        T = len(observations)
        ll, alphas = self.alpha_pass(observations)
        _, betas = self.beta_pass(observations)

        gammas = []
        digammas = []
        # We need one digamma for every t
        for t in range(T - 1):
            digamma = self.A.hadamard(
                alphas[t].outer(
                    self.B.get_col(observations[t + 1]).hadamard(betas[t + 1])
                )
            )
            digamma /= ll
            digammas.append(digamma)
            gammas.append(digamma.sum_rows())

            # temp_matrix = alphas[t].outer(self.B.get_col(observations[t + 1]).hadamard(betas[t + 1]))
            # curr_digamma = self.A.hadamard(temp_matrix).apply_func(lambda x: x / ll)
            # digammas.append(curr_digamma)
            # gammas.append(curr_digamma.sum_rows())
            # assert all([[abs(c1 - c2) < 1e-10 for c1, c2 in zip(r1, r2)]
            #             for r1, r2 in zip(digamma.data, curr_digamma.data)])

        # Add last gamma for time step T
        gammas.append(alphas[-1] / ll)
        return gammas, digammas

    def reestimate(self, observations: list, gammas: List[Vector],
                   digammas: List[Matrix2d]) -> Tuple[Vector, Matrix2d, Matrix2d]:
        """
        Implementation of Baum-Welch algorithm's parameters re-estimation using computed gammas and digammas.
        :param list observations: {Ot} for t=1...T, where Ot in {0, ..., K}
        :param list gammas: computed from gamma_pass()
        :param list digammas: computed from gamma_pass()
        :return: a tuple containing the new pi, A, B as Vector, Matrix2d, Matrix2d objects respectively
        """
        # Calculate new pi
        new_pi = gammas[0].normalize()
        # SUM all digammas
        # Calculate new transition matrix (A)
        digammas_sum = [
            [sum(dcolumn) for dcolumn in zip(*drows)]
            for drows in zip(*digammas)]
        # Sum all gammas
        gammas_sum = [sum(dcolumn) for dcolumn in zip(*gammas[:-1])]
        new_A = Matrix2d([[col / to_divide for col in row]
                          for row, to_divide in zip(digammas_sum, gammas_sum)]).normalize_rows()

        # Calculate new observation matrix (B)
        # Need a mapping from observation to time steps

        # Can this be done with list comprehensions?
        o2t = [[0, ] for _ in range(self.K)]
        for t, o in enumerate(observations):
            o2t[o].append(t)
        # TODO utilize previous calc for the previous sum
        # Sum all gammas
        gammas_sum = [sum(dcolumn) for dcolumn in zip(*gammas)]
        # New_B is NxK
        # See stamp tutorial for an explanation of notation
        new_B = Matrix2d([
            [sum([gammas_j[t] for t in t_s]) for t_s in o2t]
            for gamma_sum_j, gammas_j in zip(gammas_sum, zip(*gammas))
        ]).normalize_rows()
        return new_pi, new_A, new_B

    def delta_pass(self, observations: list) -> Tuple[Vector, float]:
        """
        Implementation of Viterbi algorithm to compute the most probable state sequence for the given observation
        sequence.
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
    #   - output most probable states path
    _states_path, _ = _hmm.delta_pass(_observations)
    print(_states_path.__str__(include_shape=False))
