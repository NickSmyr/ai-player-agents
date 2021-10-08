import fileinput
from typing import Tuple

import numpy as np

from hmm import HMM
from hmm_utils import Vector, Matrix2d


def close(t1, t2, tol: float):
    return np.allclose(np.array(t1), np.array(t2), rtol=tol, atol=tol)


def get_gt() -> Tuple[Matrix2d, Matrix2d, Vector]:
    A_gt = Matrix2d([
        [.70, .05, .25],
        [.10, .80, .10],
        [.20, .30, .50],
    ])
    B_gt = Matrix2d([
        [.70, .20, .10, .00],
        [.10, .40, .30, .20],
        [.00, .10, .20, .70],
    ])
    pi_gt = Vector(
        [1.00, 0.00, 0.00]
    )
    return A_gt, B_gt, pi_gt


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


def q7(hmm: HMM, observations: list, p_tol: float = 1e-6, max_iter: int = 100) -> None:
    # Initialize model
    A_q7, B_q7, pi_q7 = get_q7()
    hmm.initialize_static(A=A_q7, B=B_q7, pi=pi_q7)
    # Output best model estimate
    A, B, pi = hmm.train(observations, p_tol=p_tol, max_iter=max_iter)
    last_i, last_ll = hmm.last_i, hmm.last_ll
    print(f'[Q7][N={len(observations)}] i={last_i:02d} | log(p)={last_ll:.5f}')
    print(f'[Q7][N={len(observations)}] A={str(A)}')
    print(f'[Q7][N={len(observations)}] B={str(B)}')
    print(f'[Q7][N={len(observations)}] pi={str(pi)}')


if __name__ == '__main__':
    _tol = 1e-2
    for _sample_index in [1000, 10000]:
        #   - read files
        _input = fileinput.input(f'hmmc_sample_{_sample_index}.in')
        #   - initialize HMMs
        _hmm, _observations = HMM.from_input(_input)
        _input.close()

        # Question 7
        q7(hmm=_hmm, observations=_observations, max_iter=100, p_tol=1e-6)

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
