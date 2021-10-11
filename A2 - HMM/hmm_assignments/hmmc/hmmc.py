import fileinput
import math
import random
import sys
from typing import Tuple, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from hmm import HMM
from hmm_utils import Vector, Matrix2d

eps = sys.float_info.epsilon
matplotlib.rcParams["font.family"] = 'JetBrains Mono'


def close(t1, t2, tol: float):
    return np.allclose(np.array(t1), np.array(t2), rtol=tol, atol=tol)


def plot_learning_curves(q_index: int, A_diff, B_diff, pi_diff, max_iters: int, title: str, last_i: int,
                         last_ll: float, secondary: bool = False, show: bool = True) -> None:
    colors = {
        'r': '#D32F2F' if not secondary else '#FFCDD2',
        'g': '#388E3C' if not secondary else '#C8E6C9',
        'b': '#303F9F' if not secondary else '#C5CAE9',
    }
    suffix = 'gt' if not secondary else 'init'
    plt.plot(list(range(last_i)), A_diff[:last_i], '-+', label=f'|A - A_{suffix}|', color=colors['r'].lower())
    plt.plot(list(range(last_i)), B_diff[:last_i], '-+', label=f'|B - B_{suffix}|', color=colors['g'].lower())
    plt.plot(list(range(last_i)), pi_diff[:last_i], '-+', label=f'|pi - pi_{suffix}|', color=colors['b'].lower())
    plt.xlabel('i (iteration index)')
    plt.ylabel('|diff_i| (diff from ground truth)')
    plt.title(f'[Q{q_index}] {title.replace("=", "_")} '
              f'(max_iters={max_iters}, last_i={last_i}, last_ll={last_ll:.3f})')
    if show:
        plt.legend()
        plt.savefig(f'q{q_index}_{title.lower()}.svg')
        plt.show()


# ---------------------------------------------------------------------------------------------- #
# -------------------------->         Matrix Initializations         <-------------------------- #
# ---------------------------------------------------------------------------------------------- #


def get_q7() -> Tuple[Matrix2d, Matrix2d, Vector]:
    A_q7 = Matrix2d([
        [.54, .26, .20],
        [.19, .53, .28],
        [.22, .18, .60],
    ])
    B_q7 = Matrix2d([
        [.50, .20, .11, .19],
        [.22, .28, .23, .27],
        [.19, .21, .45, .15],
    ])
    pi_q7 = Vector(
        [.30, 0.20, 0.50]
    )
    return A_q7, B_q7, pi_q7


A8, B8, pi8 = None, None, None


def get_q8() -> Tuple[Matrix2d, Matrix2d, Vector]:
    N, K = 3, 4
    global A8, B8, pi8
    if A8 is None:
        A8, B8, pi8 = Matrix2d.random(N, N), Matrix2d.random(N, K), Vector.random(N, normalize=True)
    return A8, B8, pi8


def get_q10a(**kwargs) -> Tuple[Matrix2d, Matrix2d, Vector]:
    N, K = 3, 4
    A_q10 = Matrix2d([[(1. + 0.0001 * random.random()) / N] * N] * N)
    B_q10 = Matrix2d([[(1. + 0.0001 * random.random()) / K] * K] * N)
    pi_q10 = Vector([1. / N] * N)
    return A_q10.normalize_rows(), B_q10.normalize_rows(), pi_q10.normalize()


def get_q10b(**kwargs) -> Tuple[Matrix2d, Matrix2d, Vector]:
    N, K = 3, 4
    # A identity matrix
    A_q10 = Matrix2d([[1.0 if j == i else eps for j in range(N)] for i in range(N)])
    # B uniform
    B_q10 = Matrix2d([[1.0 if j == i else eps for j in range(K)] for i in range(N)])
    # pi [0, 0, 1]
    pi_q10 = Vector([eps, eps, 1.])
    return A_q10.normalize_rows(), B_q10.normalize_rows(), pi_q10.normalize()


def get_q10c(hmm_gt: HMM) -> Tuple[Matrix2d, Matrix2d, Vector]:
    N, K = 3, 4
    A_gt, B_gt, pi_gt = hmm_gt.A.copy(), hmm_gt.B.copy(), hmm_gt.pi.copy()
    # A identity matrix
    A_q10 = Matrix2d([[(A_gt[i][j] + 0.05 * random.random()) for j in range(N)] for i in range(N)])
    # B uniform
    B_q10 = Matrix2d([[(B_gt[i][j] + 0.05 * random.random()) for j in range(K)] for i in range(N)])
    # pi [0, 0, 1]
    pi_q10 = Vector([(pi_j + 0.05 * random.random()) for pi_j in pi_gt])
    return A_q10.normalize_rows(), B_q10.normalize_rows(), pi_q10.normalize()


# ---------------------------------------------------------------------------------------------- #
# ------------------------------>         Questions         <----------------------------------- #
# ---------------------------------------------------------------------------------------------- #


def q7(hmm: HMM, hmm_gt: HMM, observations: list, p_tol: float = 1e-6, max_iters: int = 100, title: str = '') -> None:
    # Initialize model
    A_q7, B_q7, pi_q7 = get_q7()
    hmm.initialize_static(A=A_q7, B=B_q7, pi=pi_q7)
    # Output best model estimate
    A, B, pi, A_diff, B_diff, pi_diff, A_diff_init, B_diff_init, pi_diff_init = \
        hmm.train(observations, p_tol=p_tol, max_iters=max_iters, hmm_gt=hmm_gt, dist='l1')
    last_i, last_ll = hmm.last_i, hmm.last_ll
    print(f'[Q7][N={len(observations)}] i={last_i:02d} | log(p)={last_ll:.5f}')
    print(f'[Q7][N={len(observations)}] A={str(A)}')
    print(f'[Q7][N={len(observations)}] B={str(B)}')
    print(f'[Q7][N={len(observations)}] B={str(pi)}')
    # Plot learning curves
    plot_learning_curves(7, A_diff, B_diff, pi_diff, title=title, max_iters=max_iters, last_i=hmm.last_i,
                         last_ll=hmm.last_ll, show=False)
    plot_learning_curves(7, A_diff_init, B_diff_init, pi_diff_init, title=title, max_iters=max_iters, last_i=hmm.last_i,
                         last_ll=hmm.last_ll, secondary=True)


def q8(hmm: HMM, hmm_gt: HMM, observations: list, p_tol: float = 1e-6, max_iters: int = 100, title: str = '') -> None:
    # Initialize model
    A_q8, B_q8, pi_q8 = get_q8()
    # print(A_q8.__str__(round_places=4, include_shape=False))
    # print(B_q8.__str__(round_places=4, include_shape=False))
    # print(pi_q8.__str__(round_places=4, include_shape=False))

    hmm.initialize_static(A=A_q8, B=B_q8, pi=pi_q8)
    # Output best model estimate
    A, B, pi, A_diff, B_diff, pi_diff, A_diff_init, B_diff_init, pi_diff_init = \
        hmm.train(observations, p_tol=p_tol, max_iters=max_iters, hmm_gt=hmm_gt, dist='l1')
    last_i, last_ll = hmm.last_i, hmm.last_ll
    print(f'[Q8][N={len(observations)}] i={last_i:02d} | log(p)={last_ll:.5f}')
    # Plot learning curves
    plot_learning_curves(8, A_diff, B_diff, pi_diff, title=title, max_iters=max_iters, last_i=hmm.last_i,
                         last_ll=hmm.last_ll, show=False)
    plot_learning_curves(8, A_diff_init, B_diff_init, pi_diff_init, title=title, max_iters=max_iters, last_i=hmm.last_i,
                         last_ll=hmm.last_ll, secondary=True)
    # Plot final error
    print(f'A_diff[-1]={A_diff[-1]}, B_diff[-1]={B_diff[-1]}, pi_diff[-1]={pi_diff[-1]}')


def q9(observations: list, p_tol: float = 1e-6, max_iters: int = 100) -> list:
    N_RUNS = 10
    N_STATES = [1, 2, 3, 5, 10]
    avg_lls = [0.] * len(N_STATES)

    for Ni, N in enumerate(N_STATES):
        print(f'[q9] N={N}')
        for ri in range(N_RUNS):
            #   - initialize an HMM with the given number of hidden states
            hmm = HMM(N=N, K=4)
            hmm.initialize()
            #   - train it on the given observation sequence
            hmm.train(observations=observations, max_iters=max_iters, p_tol=p_tol)
            #   - get last sequence evaluation
            avg_lls[Ni] += hmm.last_ll / N_RUNS
            print(f'[q9] N={N} > r={ri} [DONE]')

    # Return average log-likelihood of the observation in each of the different HMMs
    return avg_lls


def q10(hmm_gt: HMM, observations: list, p_tol: float = 1e-6, max_iters: int = 100) -> Tuple[List[int], List[float]]:
    last_is = [-1] * 3
    last_lls = [-math.inf] * 3
    for qi, q in enumerate(['a', 'b', 'c']):
        #   - get matrices
        A_q, B_q, pi_q = globals()[f'get_q10{q}'](hmm_gt=hmm_gt)
        # print(A_q)
        # print(B_q)
        # print(pi_q)
        #   - initialize model
        hmm = HMM(N=3, K=4)
        hmm.initialize_static(A=A_q, B=B_q, pi=pi_q)
        #   - train model on the given observation sequence
        hmm.train(observations=observations, max_iters=max_iters, p_tol=p_tol)
        last_is[qi], last_lls[qi] = hmm.last_i, hmm.last_ll
        print(f'[q={q}] last_is[qi]={last_is[qi]}, last_lls[qi]={last_lls[qi]:.4f}')
    return last_is, last_lls


# ---------------------------------------------------------------------------------------------- #
# ----------------------------------->         Main         <----------------------------------- #
# ---------------------------------------------------------------------------------------------- #


if __name__ == '__main__':
    _tol = 1e-6
    for _sample_index in [1000, 10000]:
        #   - read files
        _input = fileinput.input(f'hmmc_sample_{_sample_index}.in')
        _output = fileinput.input(f'hmmc_sample_{_sample_index}.out')
        #   - get ground truth HMM
        _hmm_gt, _ = HMM.from_input(_output)
        _output.close()
        #   - initialize HMMs
        _hmm, _observations = HMM.from_input(_input)
        _input.close()

        # # Question 7
        # q7(hmm=_hmm, hmm_gt=_hmm_gt, observations=_observations, max_iters=500, p_tol=1e-3,
        #    title=f'N={_sample_index}')

        # # Question 8
        # q8(hmm=_hmm, hmm_gt=_hmm_gt, observations=_observations, max_iters=500, p_tol=_tol,
        #    title=f'N={_sample_index}')

        # # Question 9
        # _avg_lls = q9(observations=_observations, max_iters=500, p_tol=1e-6)
        # print([f'{_ll:.4f}' for _ll in _avg_lls])

        # Question 10
        _last_is, _last_lls = q10(hmm_gt=_hmm_gt, observations=_observations, max_iters=500, p_tol=1e-4)
        print(f'[Q10][N={_sample_index}] last_is={_last_is}')
        print(f'[Q10][N={_sample_index}] last_lls={_last_lls}')
