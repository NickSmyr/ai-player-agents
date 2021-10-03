#!/usr/bin/env python3

import fileinput
import math
from typing import Optional, Tuple, List

import numpy as np


def np2str(t: np.ndarray, round_places: int = -1, include_shape: bool = True) -> str:
    shape_str = t.shape.__str__() + ' ' if include_shape else ''
    if round_places == -1:
        return (shape_str + t.__str__()).replace('\n', '') \
            .replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace(',', '').replace('  ', ' ')
    data_str = ''
    for d in t:
        if type(d) == list:
            for dd in d:
                data_str += str(round(dd, round_places)) + ' '
        else:
            data_str += str(round(d, round_places)) + ' '
    return (shape_str + data_str.rstrip()).replace('[', '').replace(']', '').replace('(', '').replace(')', '') \
        .replace(',', '').replace('\'', '').replace('\n', '').replace('  ', ' ')

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
    Our implementation of a 1-st order Hidden Markov Model USING NUMPY.
    """

    def __init__(self, N: int, K: int, A: Optional[np.ndarray] = None, B: Optional[np.ndarray] = None,
                 pi: Optional[np.ndarray] = None):
        """
        HMM class constructor.
        :param int N: number of hidden states
        :param int K: number of emission types
        :param (optional) A: the transmission model matrix
        :param (optional) B: the observation model matrix
        :param (optional) pi: the initial states pfm
        """
        # Random intialization of parameters if not provided
        if A is None:
            A = np.random.rand(N, N)
            A /= np.sum(A, axis=1, keepdims=True)
        self.A = A
        if B is None:
            B = np.random.rand(N, K)
            B /= np.sum(B, axis=1, keepdims=True)
        self.B = B
        if pi is None:
            pi = np.random.rand(1, N)
            pi /= np.sum(pi)
        self.pi = pi
        self.N = N
        self.K = K

    def alpha_pass(self, observations: np.ndarray,
                   T: Optional[int] = None) -> Tuple[float, np.ndarray or None, List[float]]:
        """
        Perform a forward pass to compute the likelihood of an observation sequence.
        :param list observations: {Ot} for t=1...T, where Ot in {0, ..., K}
        :param (optional) T: total number of observations (in case list was pre-initialized but not fully filled)
        :return: a tuple object containing the (marginalized) probability that the HMM emitted the given observations,
                 a list of all the alphas computed during the recursive computation (as np.ndarray objects) and
                 a list of all c's (aka the scaling coefficients)
        """
        T = len(observations) if T is None else T
        if T == 1:
            return 0., None, []
        # Initialize alpha
        alpha_tm1 = np.multiply(self.pi, self.B[:, observations[0]])
        c = 1 / alpha_tm1.sum()
        # Store alphas (and Cs) in memory
        cs = [c, ]
        #   - scale a_0
        alpha_tm1 *= c
        alphas = np.zeros((T, self.N), dtype=float)
        alphas[0, :] = alpha_tm1
        # Perform alpha-pass iterations
        for t in range(1, T):
            #   - compute alpha
            alpha = np.multiply(np.dot(alpha_tm1, self.A), self.B[:, observations[t]])
            #   - scale alpha
            c = 1 / alpha.sum()
            alpha *= c
            #   - check for underflow
            if c == 0.0:
                raise RuntimeError(f'll drove to 0.0 (t={t})')
            #   - append to list
            alphas[t, :] = alpha
            cs.append(c)
            alpha_tm1 = alpha
        # Return likelihood (sum of last alpha vec) and the recorded alphas
        return -sum([math.log10(c) for c in cs]), alphas, cs

    def beta_pass(self, observations: np.ndarray, cs: Optional[List] = None,
                  T: Optional[int] = None) -> Tuple[float, Optional[np.ndarray]]:
        """
        Perform a backward pass as another way to compute the likelihood of an observation sequence.
        :param list observations: {Ot} for t=1...T, where Ot in {0, ..., K}
        :param (optional) cs: {Ct} for t=1...T, where Ct is the scaling coefficient for alpha_t
        :param (optional) T: total number of observations (in case list was pre-initialized but not fully filled)
        :return: a tuple object containing the (marginalized) probability that the HMM emitted the given observations
                 and a list of all the betas computed during the recursive computation (as np.ndarray objects)
        """
        T = observations.shape[0] if T is None else T
        if T == 1:
            return 0., None
        # Initial beta is beta_{T-1}
        if cs is None:
            cs = [1.] * T
        betas = np.zeros((T, self.N), dtype=float)
        beta_tp1 = np.array([cs[-1]] * self.N, dtype=float)
        betas[T - 1, :] = beta_tp1
        # Iterate through reversed time
        for t in range(T - 2, -1, -1):
            #   - compute beta_t
            beta_t = np.dot(self.A, np.multiply(self.B[:, observations[t + 1]], beta_tp1))
            #   - scale beta_t[i]
            beta_t *= cs[t]
            #   - append to betas list
            betas[t, :] = beta_t
            #   - save for next iteration
            beta_tp1 = beta_t
        # Return likelihood, betas
        return betas[0, :].sum(), betas

    def gamma_pass(self, observations: np.ndarray, alphas: Optional[List[np.ndarray]] = None, cs: Optional[list] = None,
                   T: Optional[int] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Implementation of Baum-Welch algorithm's Gamma Pass to compute gammas & digammas (i.e. prob of being in state i
        at time t and moving to state j at time t+1).
        :param list observations: {Ot} for t=1...T, where Ot in {0, ..., K}
        :param (optional) alphas: output from alpha_pass
        :param (optional) cs: output from alpha_pass
        :param (optional) T: total number of observations (in case list was pre-initialized but not fully filled)
        :return: a tuple containing ([Gamma_t = np.ndarray(P(Xt=i|O_1:t,HMM) for all i's) for t = 1...T],
                                     [DiGamma_t = np.ndarray(P(Xt=i,Xt+1=j|O_1:t,HMM) for all (i, j)'s) for t = 1...T])
        """
        T = len(observations) if T is None else T
        if T == 1:
            return [], []
        # 1. Get alpha_t and beta_t for all t=0,...,T-1
        if alphas is None:
            _, alphas, cs = self.alpha_pass(observations)
        _, betas = self.beta_pass(observations, cs=cs)
        # 2. Compute digammas and gammas for every t
        gammas, digammas = [], []
        for t in range(T - 1):
            #   - compute digamma_t
            digamma = np.multiply(self.A, np.outer(alphas[t],
                                                   np.multiply(self.B[:, observations[t + 1]], betas[t + 1])))
            digammas.append(digamma)
            #   - marginalize over i (rows) to compute gamma_t
            gammas.append(np.sum(digamma, axis=1))
        # Add last gamma for time step T
        gammas.append(alphas[-1])
        return gammas, digammas

    def reestimate(self, observations: np.ndarray, gammas: List[np.ndarray], digammas: List[np.ndarray],
                   lambda_mix: float = 1.15, T: Optional[int] = None) -> None:
        """
        Implementation of Baum-Welch algorithm's parameters re-estimation using computed gammas and digammas.
        Source: A Revealing Introduction to Hidden Markov Models, Mark Stamp
        :param list observations: {Ot} for t=1...T, where Ot in {0, ..., K}
        :param list gammas: computed from gamma_pass()
        :param list digammas: computed from gamma_pass()
        :param float lambda_mix: model averaging weight (A = newA * lambda_mix + A * (1.0-lambda_mix)
        :param (optional) T: total number of observations (in case list was pre-initialized but not fully filled)
        :return: a tuple containing the new pi, A, B all as np.ndarray objects
        """
        T = len(observations) if T is None else T
        if T == 1:
            return
        rN = range(self.N)
        # Reestimate pi
        self.pi = self.pi * (1.0 - lambda_mix) + gammas[0] * lambda_mix
        for i in rN:
            denom = 0.
            for t in range(T - 1):
                denom += gammas[t][i]
            # Reestimate A
            for j in rN:
                numer = 0.
                for t in range(T - 1):
                    numer += digammas[t][i, j]
                if denom != 0.0:
                    self.A[i, j] = (1.0 - lambda_mix) * self.A[i, j] + lambda_mix * (numer / denom)
            # Reestimate B
            denom += gammas[T - 1][i]
            for j in range(self.K):
                numer = 0.
                for t in range(T):
                    if observations[t] == j:
                        numer += gammas[t][i]
                if denom != 0.0:
                    self.B[i, j] = (1.0 - lambda_mix) * self.B[i, j] + lambda_mix * (numer / denom)
        # Normalize model parameters and return
        self.pi /= np.sum(self.pi)
        self.A /= np.sum(self.A, axis=1, keepdims=True)
        self.B /= np.sum(self.B, axis=1, keepdims=True)

    def baum_welch(self, observations: np.ndarray, tol: float = 1e-5, max_iter: int = 100) -> Optional[float]:
        """
        Implementation of Baum-Welch algorithm to estimate optimal HMM params (A, B) using successive gamma passes and
        re-estimation, until observations likelihood (alpha_pass) converges.
        :param list observations: {Ot} for t=1...T, where Ot in {0, ..., K}
        :param float tol: convergence criterion
        :param int max_iter: maximum allowed number of iterations
        :return: the final_likelihood as a float
        """
        T = observations.shape[0]
        # Get initial likelihood
        old_ll, alphas, cs = self.alpha_pass(observations=observations)
        # Successively improve it
        ll = None
        for i in range(max_iter):
            #   - compute gammas
            gammas, digammas = self.gamma_pass(observations=observations, alphas=alphas, cs=cs, T=T)
            #   - reestimate model parameters and update model
            self.reestimate(observations=observations, gammas=gammas, digammas=digammas, T=T)
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
        return ll

    def delta_pass(self, observations: list) -> Tuple[np.ndarray, float]:
        """
        Implementation of Viterbi algorithm to compute the most probable state sequence for the given observation
        sequence.
        [Theory] HMMs can be used to maximize the expected number of correct states.
        [  >>  ] DP can be used to maximize the entire sequence probability.
        :param list observations: {Ot} for t=1...T, where Ot in {0, ..., K}
        :return: a tuple containing the most probable state sequence in a np.ndarray object and the probability of
                 the most probable path as float object
        """
        delta = np.log10(self.pi).T + np.log10(self.B[:, observations[0]])
        # deltas = [delta, ]
        deltas_argmax = []
        for t in range(1, len(observations)):
            delta_data = [argmax(delta + np.log10(self.A[:, i])) for i in range(self.N)]
            delta = np.array([t[0] for t in delta_data]) + np.log10(self.B[:, observations[t]])
            delta_argmax = np.array([t[1] for t in delta_data])
            # deltas.append(delta)
            deltas_argmax.append(delta_argmax)
        # Calculate states path
        states_path_prob, last_state_index = np.max(delta), np.argmax(delta)
        states_path = [last_state_index, ]
        for i in range(len(deltas_argmax) - 1, -1, -1):
            # states_path.append(deltas[i].argmax_data[states_path[-1]])
            states_path.append(deltas_argmax[i][states_path[-1]])
        states_path.reverse()
        return np.array(states_path).dtype(int), states_path_prob

    @staticmethod
    def line2list2(line: str):
        line_data = [x for x in line.rstrip().split(" ")]
        nrows = int(line_data.pop(0))
        ncols = int(line_data.pop(0))
        assert nrows * ncols == len(line_data), f'Given numbers of elements do not match ' \
                                                f'((nrows,ncols)={(nrows, ncols)} | len(line_data)={len(line_data)})'
        return [[float(line_data[j + i * ncols]) for j in range(ncols)] for i in range(nrows)]

    @staticmethod
    def line2list1(line: str) -> list:
        line_data = [x for x in line.rstrip().split(" ")]
        nrows = int(line_data.pop(0))
        assert nrows == 1, f'Vector should be row-vectors (nrows={nrows})'
        ncols = int(line_data.pop(0))
        assert ncols == len(line_data), f'Given numbers of elements do not match (ncols={ncols} | ' \
                                        f'len(line_data)={len(line_data)})'
        return [float(lj) for lj in line_data]

    # noinspection PyProtectedMember
    @staticmethod
    def from_input(finput: fileinput.FileInput) -> Tuple['HMM', Optional[np.ndarray]]:
        """
        Initialize a new HMM instance using input provided in the form of Kattis text files.
        :param FileInput finput: a FileInput instance instantiated from stdin (e.g. by running python file using input
                                 redirect from a *.in file)
        :return: an HMM instance
        """
        A, B, pi, N, K, obs = None, None, None, None, None, None
        for i, line in enumerate(finput):
            if i == 0:
                A = np.array(HMM.line2list2(line), dtype=float)
            elif i == 1:
                B = np.array(HMM.line2list2(line), dtype=float)
                N, K = B.shape
            elif i == 2:
                pi = np.array(HMM.line2list1(line), dtype=float)
            else:
                obs = np.array(HMM.line2list1('1 ' + line), dtype=int)
        return HMM(N=N, K=K, A=A, B=B, pi=pi), obs


if __name__ == '__main__':
    #   - initialize HMM
    _hmm, _observations = HMM.from_input(fileinput.input())
    #   - output best model estimate
    _ = _hmm.baum_welch(_observations, tol=1e-10, max_iter=50)
    print(np2str(_hmm.A))
    print(np2str(_hmm.B))
