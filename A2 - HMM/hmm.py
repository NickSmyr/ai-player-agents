# from sys import stderr
from typing import List, Tuple

from hmm_utils import Matrix2d, Vector, outer_product


class HMM:

    def __init__(self, N: int, K: int):
        # Random intialization
        # TODO: better initialization than uniform
        self.N = N
        self.K = K

        self.A = Matrix2d.random(N, N, row_stochastic=True)
        self.B = Matrix2d.random(N, K, row_stochastic=True)
        self.pi = Vector.random(N, normalize=True)

        self.A_transposed = self.A.T

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
        Perform a forward pass echoing the probability of an observation sequence.
        :param list observations: {Ot} for t=1...T, where Ot in {0, ..., K}
        :return: a float object containing the (marginalized) probability that the HMM emitted the given observations
        """
        # Initialize alpha
        alpha = self.pi.hadamard(self.B.get_col(observations[0]))
        # Store alphas in memory
        alphas = [alpha]
        # print(alpha, alpha.sum(), file=stderr)
        # Perform alpha-pass iterations
        for t in range(1, len(observations)):
            alpha = (self.A_transposed @ alpha).hadamard(self.B.get_col(observations[t]))
            alphas.append(alpha)
            # print(alpha, alpha.sum(), file=stderr)
        # Return likelihood (sum of last alpha vec)
        likelihood = alpha.sum()
        return likelihood, alphas

    def beta_pass(self, observations: list) -> Tuple[float, List[Vector]]:
        T = len(observations)
        # Initial beta is beta_{T-1}
        beta = Vector([1.] * self.A.shape[0])
        betas = [beta]
        t = T - 2
        while t > -1:
            beta = (self.A) @ (self.B.get_col(observations[t + 1]).hadamard(beta))
            betas.append(beta)
            t -= 1
        # beta_{-1} Used only for testing purposes
        betas.append(beta.hadamard(self.pi).hadamard(self.B.get_col(observations[0])))
        # Betas are ordered in reverse to match scientific notations (betas[t] is really betas[t])
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
            temp_matrix = outer_product(alphas[t], self.B.get_col(observations[t + 1]).hadamard(betas[t + 1]))
            curr_digamma = self.A.hadamard(temp_matrix).apply_func(lambda x: x / ll)
            digammas.append(curr_digamma)

            gammas.append(curr_digamma.sum_rows())
        return gammas, digammas

    def get_new_parameters(self, observations, gammas, digammas) -> Tuple[List, List, List]:
        T = len(observations)
        # Calculate new pi
        new_pi = gammas[0]
        # SUM all digammas
        # Calculate new transition matrix (A)
        digammas_sum = [
            [sum(dcolumn)
             for dcolumn in zip(*drows)]
            for drows in zip(*digammas)]
        # Sum all gammas
        gammas_sum = [sum(dcolumn) for dcolumn in zip(*gammas[:-1])]
        new_A = [ [ col / to_divide for col in row]
            for row, to_divide in zip(digammas_sum,gammas_sum) ]

        # Calculate new observation matrix (B)
        # Need a mapping from observation to time steps

        # Can this be done with list comprehensions?
        o2t = [[] for i in range(self.K)]
        for t, o in enumerate(observations):
            o2t[o].append(t)
        # TODO utilize previous calc for the previous sum
        # Sum all gammas
        gammas_sum = [sum(dcolumn) for dcolumn in zip(*gammas)]
        # New_B is NxK
        # See stamp tutorial for an explanation of notation
        new_B = [
            [ sum([ gammas_j[t] for t in t_s]) for t_s in o2t]
                for gamma_sum_j, gammas_j in zip(gammas_sum, zip(*gammas))
        ]
        return new_pi, new_A, new_B




    def delta_pass(self, observations: list) -> list:
        """
        Perform a backward pass echoing the most probable state sequence for the given observation sequence.
        :param list observations: {Ot} for t=1...T, where Ot in {0, ..., K}
        :return:
        """
        raise NotImplementedError


if __name__ == '__main__':
    hmm = HMM(N=4, K=3)
    observations = [0, 1, 2, 1]
    # Likelihood
    ll_a, _ = hmm.alpha_pass(observations)
    ll_b, _ = hmm.beta_pass(observations)
    # Approximate equality
    assert abs(ll_a - ll_b) < 1e-3, "Likelihoods don't match"

    gammas, digammas = hmm.gamma_pass(observations)
    hmm.get_new_parameters(observations, gammas, digammas)

