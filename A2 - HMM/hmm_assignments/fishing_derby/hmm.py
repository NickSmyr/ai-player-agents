import fileinput
import math
import sys
from typing import List, Tuple, Optional

from hmm_utils import Matrix2d, Vector, DeltaVector, argmax

eps = sys.float_info.epsilon


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
