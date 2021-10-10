import fileinput
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from hmm import HMM
from hmm_utils import Vector, Matrix2d

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


def get_q8() -> Tuple[Matrix2d, Matrix2d, Vector]:
    N, K = 3, 4
    return Matrix2d.random(N, N), Matrix2d.random(N, K), Vector.random(N, normalize=True)


def get_q10a() -> Tuple[Matrix2d, Matrix2d, Vector]:
    N, K = 3, 4
    return Matrix2d.random(N, N), Matrix2d.random(N, K), Vector.random(N, normalize=True)


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
    A_q7, B_q7, pi_q7 = get_q8()
    hmm.initialize_static(A=A_q7, B=B_q7, pi=pi_q7)
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


def q10(hmm: HMM, hmm_gt: HMM, observations: list, p_tol: float = 1e-6, max_iters: int = 100, title: str = '') -> None:
    # TODO
    pass


# ---------------------------------------------------------------------------------------------- #
# ----------------------------------->         Main         <----------------------------------- #
# ---------------------------------------------------------------------------------------------- #


if __name__ == '__main__':
    _tol = 1e-2
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

        # _observations = _observations[:1000]

        # Question 7
        q7(hmm=_hmm, hmm_gt=_hmm_gt, observations=_observations, max_iters=500, p_tol=1e-6, title=f'N={_sample_index}')

        # Question 8
        q8(hmm=_hmm, hmm_gt=_hmm_gt, observations=_observations, max_iters=500, p_tol=1e-6, title=f'N={_sample_index}')

        # Question 10
        q10(hmm=_hmm, hmm_gt=_hmm_gt, observations=_observations, max_iters=100, p_tol=1e-6, title=f'N={_sample_index}')

        # break

        # A_gt, B_gt, pi_gt = get_gt()
        # print(str(A_gt))
        # print(str(B_gt))
        # print(str(pi_gt))

        #   - compare output to ground truth
        #   - (a) Assert Equal to the Ground Truth
        # assert A_opt.__str__(round_places=6) == _hmm_gt.A.__str__(round_places=6), \
        #     f'{A_opt.__str__(round_places=6)} vs {_hmm_gt.A.__str__(round_places=6)}'
        # assert B_opt.__str__(round_places=6) == _hmm_gt.B.__str__(round_places=6), \
        #     f'{B_opt.__str__(round_places=6)} vs {_hmm_gt.B.__str__(round_places=6)}'
        # #   - (b) Assert Close to the Ground Truth
        # assert close(_hmm.A, _hmm_gt.A, tol=_tol), f'A_opt not close to _hmm_gt.A for tol={_tol}'
        # assert close(_hmm.B, _hmm_gt.B, tol=_tol), f'A_opt not close to _hmm_gt.A for tol={_tol}'
        # print(f'#{_sample_index:02d} PASSed', file=stderr)
