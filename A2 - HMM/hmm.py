# from sys import stderr
from fileinput import FileInput
from typing import List, Tuple, Optional

from hmm_utils import Matrix2d, Vector, DeltaVector, argmax


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
        # TODO: better initialization than uniform
        self.A = Matrix2d.random(N, N, row_stochastic=True) if A is None else A
        self.B = Matrix2d.random(N, K, row_stochastic=True) if B is None else B
        self.pi = Vector.random(N, normalize=True) if pi is None else pi
        self.A_transposed = self.A.T

        self.N = N
        self.K = K

        # ------------------------------------
        # Shape Tests:
        # ------------------------------------
        # C = (self.A @ self.B).T @ self.A.T
        # print(type(C), C.shape)
        # Cv = C @ Vector.random(4)
        # print(type(Cv), Cv.shape)
        # ------------------------------------
        # All passed.
        # ------------------------------------

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
            t -= 1
        # beta_{-1} Used only for testing purposes
        betas.append(beta.hadamard(self.pi).hadamard(self.B.get_col(observations[0])))
        # Betas are ordered in reverse to match scientific notations (betas[t] is really beta_t)
        betas.reverse()
        # Return likelihood, betas
        return betas[0].sum(), betas

    def gamma_pass(self, observations: list) -> Tuple[List[Vector], List[Matrix2d]]:
        T = len(observations)
        ll, alphas = self.alpha_pass(observations)
        _, betas = self.beta_pass(observations)

        gammas = []
        digammas = []
        # We need one digamma for every t
        for t in range(T - 1):
            temp_matrix = alphas[t].outer(self.B.get_col(observations[t + 1]).hadamard(betas[t + 1]))
            curr_digamma = self.A.hadamard(temp_matrix).apply_func(lambda x: x / ll)
            digammas.append(curr_digamma)

            gammas.append(curr_digamma.sum_rows())

        # Add last gamma for time step T-1
        gammas.append(alphas[-1])
        return gammas, digammas

    def get_new_parameters(self, observations, gammas, digammas) -> Tuple[List, List, List]:
        # Calculate new pi
        new_pi = gammas[0]
        # SUM all digammas
        # Calculate new transition matrix (A)
        digammas_sum = [
            [sum(dcolumn) for dcolumn in zip(*drows)]
            for drows in zip(*digammas)]
        # Sum all gammas
        gammas_sum = [sum(dcolumn) for dcolumn in zip(*gammas[:-1])]
        new_A = [[col / to_divide for col in row]
                 for row, to_divide in zip(digammas_sum, gammas_sum)]

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
        new_B = [
            [sum([gammas_j[t] for t in t_s]) for t_s in o2t]
            for gamma_sum_j, gammas_j in zip(gammas_sum, zip(*gammas))
        ]
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
        i = len(deltas_argmax) - 1
        while i >-1:
            # states_path.append(deltas[i].argmax_data[states_path[-1]])
            states_path.append(deltas_argmax[i][states_path[-1]])
            i -= 1
        # Reverse the states so they follow the time ordering
        states_path.reverse()
        return Vector(states_path, dtype=int), states_path_prob

    @staticmethod
    def from_input(finput: FileInput) -> 'HMM':
        """
        Initialize a new HMM instance using input provided in the form of Kattis text files.
        :param FileInput finput: a FileInput instance instantiated from stdin (e.g. by running python file using input
                                 redirect from a *.in file)
        :return: an HMM instance
        """
        A, B, pi, N, K = None, None, None, None, None
        for i, line in enumerate(finput):
            if i == 0:
                A = Matrix2d.from_str(line)
            elif i == 1:
                B = Matrix2d.from_str(line)
                N, K = B.shape
            else:
                pi = Vector.from_str(line)
        return HMM(N=N, K=K, A=A, B=B, pi=pi)

