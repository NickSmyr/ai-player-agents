from images.matrix_utils import Matrix2d, Vector


class HMM:

    def __init__(self, N: int, K: int):
        # Random intialization
        # TODO: better initialize
        self.A = Matrix2d.random(N, N)
        self.A.normalize_rows()
        self.B = Matrix2d.random(N, K)
        self.B.normalize_rows()
        self.pi = Vector.random(N)
        self.pi.normalize()

    def alpha_pass(self, O: list) -> float:
        """
        Perform a forward pass echoing the probability of an observation sequence.
        P(Ο[1:T]) = P()

        B * pi
        :param list O: {Ot} for t=1...T, where Ot in {0, ..., K}
        :return: a float object
        """
        # Initialize alpha
        alpha = self.pi @ self.B.get_col(O[0])


if __name__ == '__main__':
    HMM(N=4, K=2)
