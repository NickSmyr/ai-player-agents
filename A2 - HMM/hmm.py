# from sys import stderr

from hmm_utils import Matrix2d, Vector


class HMM:

    def __init__(self, N: int, K: int):
        # Random intialization
        # TODO: better initialization than uniform
        self.A = Matrix2d.random(N, N, row_stochastic=True)
        self.B = Matrix2d.random(N, K, row_stochastic=True)
        self.pi = Vector.random(N, normalize=True)

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

    def alpha_pass(self, observations: list) -> float:
        """
        Perform a forward pass echoing the probability of an observation sequence.
        :param list observations: {Ot} for t=1...T, where Ot in {0, ..., K}
        :return: a float object containing the (marginalized) probability that the HMM emitted the given observations
        """
        # Initialize alpha
        # t1 = float(timeit.timeit(lambda: self.pi.hadamard(self.B.get_col(observations[0])), number=1000000)) / 1000000
        # t2 = float(timeit.timeit(lambda: self.pi.hadamard(self.B.T.data[observations[0]]), number=1000000)) / 1000000
        # print(t1, t2)  # -> pretty much identical times
        alpha = self.pi.hadamard(self.B.get_col(observations[0]))
        # print(alpha, alpha.sum(), file=stderr)
        # Perform alpha-pass iterations
        for t in range(1, len(observations)):
            alpha = (self.A.T @ alpha).hadamard(self.B.get_col(observations[t]))
            # print(alpha, alpha.sum(), file=stderr)
        return alpha.sum()

    def delta_pass(self, observations: list) -> list:
        """
        Perform a backward pass echoing the most probable state sequence for the given observation sequence.
        :param list observations: {Ot} for t=1...T, where Ot in {0, ..., K}
        :return:
        """
        raise NotImplementedError


if __name__ == '__main__':
    hmm = HMM(N=4, K=2)
    hmm.alpha_pass([0, 0, 1, 0])
