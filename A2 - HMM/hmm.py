import fileinput
import math
import sys
from typing import List, Tuple, Optional

import numpy as np

from hmm_utils import Matrix2d, Vector

eps = sys.float_info.epsilon


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

    def initialize(self, globalB: Optional[list] = None, label: Optional[int] = None) -> None:
        """
        Initializes model matrices before training.
        :param list globalB: of shape (1,K)
        :param int label: e.g. the gt fish_type that this HMM instance is assumed to recognize
        """
        # Initialize B from observations
        if globalB is not None:
            B = Matrix2d([globalB.copy() for _ in range(self.N)])
        else:
            B = Matrix2d.random(nrows=self.N, ncols=self.K, row_stochastic=True)
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
        if cs is None:
            cs = [1.] * T
        beta_tp1 = Vector([cs[-1]] * self.N)
        betas = [beta_tp1, ]

        for t in range(T - 2, -1, -1):
            # O_tp1 = observations[t+1]
            #   - compute beta_t
            beta_t = A @ beta_tp1.hadamard(B_T[observations[t + 1]])
            #   - scale beta_t[i]
            beta_t *= cs[t]
            #   - append to betas list
            betas.append(beta_t)
            #   - save for next iteration
            beta_tp1 = beta_t
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
        return Vector(pi), Matrix2d(A), Matrix2d(B)

    # noinspection PyUnboundLocalVariable
    def train(self, observations: list, max_iters: int = 100, p_tol: float = 1e-6, T: Optional[int] = None,
              hmm_gt: Optional['HMM'] = None, dist: str = 'l2'):
        """
        Implementation of Baum-Welch algorithm to estimate optimal HMM params (A, B) using successive gamma passes and
        re-estimation, until observations likelihood (alpha_pass) converges.
        :param list observations: {Ot} for t=1...T, where Ot in {0, ..., K}
        :param float p_tol: convergence criterion
        :param int max_iters: maximum allowed number of iterations
        :param (optional) T: total number of time steps
        :param (optional) hmm_gt: hmm_gt to plot the learning curves
        :param str dist: matrix distance metric
        :return: a tuple object containing the (A, B, final_likelihood) matrices of the converged model
        """
        A = self.A.copy()
        A_init = A.copy()
        A_T = A.T
        B = self.B.copy()
        B_init = B.copy()
        B_T = B.T
        pi = self.pi.copy()
        pi_init = pi.copy()
        if T is None:
            T = len(observations)

        # Check for gt
        if hmm_gt is not None:
            A_gt, B_gt, pi_gt = hmm_gt.A, hmm_gt.B, hmm_gt.pi
            A_diff_gt, B_diff_gt, pi_diff_gt = np.zeros(max_iters + 1, dtype=float), \
                                               np.zeros(max_iters + 1, dtype=float), \
                                               np.zeros(max_iters + 1, dtype=float)
            A_diff_init, B_diff_init, pi_diff_init = np.zeros(max_iters + 1, dtype=float), \
                                                     np.zeros(max_iters + 1, dtype=float), \
                                                     np.zeros(max_iters + 1, dtype=float)

            def mat_diff(m1, m2) -> float:
                amb = np.abs(np.array(m1) - np.array(m2))
                if dist == 'l2':
                    return float(np.linalg.norm(amb, ord=None)) ** 2
                elif dist == 'l1':
                    return amb.sum()

            A_diff_gt[0] = mat_diff(A, A_gt)
            A_diff_init[0] = mat_diff(A, A_init)
            B_diff_gt[0] = mat_diff(B, B_gt)
            B_diff_init[0] = mat_diff(B, B_init)
            pi_diff_gt[0] = mat_diff(pi, pi_gt)
            pi_diff_init[0] = mat_diff(pi, pi_init)

        old_ll, ll, i = -math.inf, 0., 0
        for i in range(max_iters):
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
            elif ll_diff < p_tol:
                break
            A_T = A.T
            B_T = B.T
            old_ll = ll

            # Compare with gt & init
            if hmm_gt is not None:
                A_diff_gt[i + 1] = mat_diff(A, A_gt)
                B_diff_gt[i + 1] = mat_diff(B, B_gt)
                pi_diff_gt[i + 1] = mat_diff(pi, pi_gt)
                A_diff_init[i + 1] = mat_diff(A, A_init)
                B_diff_init[i + 1] = mat_diff(B, B_init)
                pi_diff_init[i + 1] = mat_diff(pi, pi_init)

        # Update parameters and return
        # noinspection PyTypeChecker
        self.initialize_static(A=A.copy(), B=B.copy(), pi=pi.copy())
        self.last_i = i + 1
        self.last_ll = ll

        if hmm_gt is None:
            return A, B, pi
        return A, B, pi, A_diff_gt, B_diff_gt, pi_diff_gt, A_diff_init, B_diff_init, pi_diff_init

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
