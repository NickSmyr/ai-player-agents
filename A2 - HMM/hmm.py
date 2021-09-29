# from sys import stderr
import fileinput
from sys import stderr
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
        self.A = A if A is not None else Matrix2d.random(N, N, row_stochastic=True)
        self.A_transposed = self.A.T
        self.B = B if B is not None else Matrix2d.random(N, K, row_stochastic=True)
        self.B_transposed = self.B.T
        self.pi = pi if pi is not None else Vector.random(N, normalize=True) if pi is None else pi

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
            #   - check for underflow
            if c == 0.0:
                raise RuntimeError(f'll drove to 0.0 (t={t})')
            #   - append to list
            alphas.append(alpha)
            cs.append(c)
            alpha_tm1 = alpha
        # Return likelihood (sum of last alpha vec) and the recorded alphas
        return -cs.log_sum(), alphas, cs

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
        # Iterate through reversed time
        for t in range(T - 2, -1, -1):
            #   - compute beta_t
            beta_t = self.A @ self.B.get_col(observations[t + 1]).hadamard(beta_tp1)
            #   - scale beta_t[i]
            beta_t *= cs[t]
            #   - append to betas list
            betas.append(beta_t)
            #   - save for next iteration
            beta_tp1 = beta_t
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
        T = len(observations)
        # 1. Get alpha_t and beta_t for all t=0,...,T-1
        if alphas is None:
            _, alphas, cs = self.alpha_pass(observations)
        _, betas = self.beta_pass(observations, cs=cs)
        # 2. Compute digammas and gammas for every t
        gammas = []
        digammas = []
        for t in range(T - 1):
            #   - compute digamma_t
            digamma = self.A.hadamard(alphas[t].outer(
                self.B.get_col(observations[t + 1]).hadamard(betas[t + 1])
            ))
            digammas.append(digamma)
            #   - marginalize over i (rows) to compute gamma_t
            gammas.append(digamma.sum_rows())
        # Add last gamma for time step T
        gammas.append(alphas[-1])
        return gammas, digammas

    def reestimate(self, observations: list, gammas: List[Vector], digammas: List[Matrix2d]) -> None:
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
        new_pi = gammas[0]
        for i in range(self.N):
            denom = 0.
            for t in range(T - 1):
                denom += gammas[t][i]
            # Reestimate A
            for j in range(self.N):
                numer = 0.
                for t in range(T - 1):
                    numer += digammas[t][i][j]
                self.A[i][j] = numer / denom
            # Reestimate B
            denom += gammas[T - 1][i]
            for j in range(self.K):
                numer = 0.
                for t in range(T):
                    if observations[t] == j:
                        numer += gammas[t][i]
                self.B[i][j] = numer / denom
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
        ll_diff = None
        # Successively improve it
        for i in range(max_iter + 1):
            if i == max_iter:
                print(f'[baum_welch] reached max_iter={max_iter} (ll_diff={ll_diff:.5f})', file=stderr)
                break
            #   - compute gammas
            gammas, digammas = self.gamma_pass(observations=observations, alphas=alphas, cs=cs)
            #   - update model
            self.reestimate(observations=observations, gammas=gammas, digammas=digammas)
            #   - check convergence
            try:
                ll, alphas, cs = self.alpha_pass(observations=observations)
                assert ll >= old_ll, f'[baum_welch] ll={ll} < old_ll={old_ll} (i={i})'
                ll_diff = ll - old_ll
                if ll_diff < 0:
                    print(f'[baum_welch] old_ll > ll (old_ll={old_ll:.5f}, ll={ll:.5f} - i={i:02d})', file=stderr)
                    break
                elif ll_diff < tol:
                    print(f'[baum_welch] reached tol={tol} at i={i} (ll={ll:.3f}, old_ll={old_ll:.3f})', file=stderr)
                    break
                else:
                    old_ll = ll
            except ZeroDivisionError as e:
                print(f'[i = {i}] ' + str(e), file=stderr)
                raise ZeroDivisionError
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
