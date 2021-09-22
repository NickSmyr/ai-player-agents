from images.matrix_utils import Matrix2d, Vector


class HMM:

    def __init__(self, N: int, K: int):
        # Random intialization
        # TODO: better initialization than uniform
        self.A = Matrix2d.random(N, N)
        self.A.normalize_rows()
        self.B = Matrix2d.random(N, K)
        self.B.normalize_rows()
        self.pi = Vector.random(N)
        self.pi.normalize()

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

    def alpha_pass(self, Ot: list) -> float:
        """
        Perform a forward pass echoing the probability of an observation sequence.
        :param list Ot: {Ot} for t=1...T, where Ot in {0, ..., K}
        :return: a float object
        """
        # Initialize alpha
        alpha = self.pi @ self.B.get_col(Ot[0])
        print(alpha)
        # Perform alpha-pass iterations
        # TODO
        return alpha


if __name__ == '__main__':
    hmm = HMM(N=4, K=2)
    hmm.alpha_pass([0, 0, 1, 0])
