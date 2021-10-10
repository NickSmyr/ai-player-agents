import fileinput
from sys import stderr

import numpy as np

# from hmm3_deliverable import HMM
from hmm import HMM
from hmm_assignments.hmmc.hmmc import plot_learning_curves


def close(t1, t2, tol: float):
    return np.allclose(np.array(t1), np.array(t2), rtol=tol, atol=tol)


if __name__ == '__main__':
    _tol = 1e-2
    for _sample_index in range(3):
        #   - read files
        _input = fileinput.input(f'hmm3_sample_{_sample_index:02d}.in')
        _output = fileinput.input(f'hmm3_sample_{_sample_index:02d}.ans')
        #   - initialize HMMs
        _hmm, _observations = HMM.from_input(_input)
        _input.close()
        _hmm_gt, _ = HMM.from_input(_output)
        _output.close()
        #   - output best model estimate
        if hasattr(_hmm, 'baum_welch'):
            _hmm.baum_welch(_observations, tol=1e-6, max_iter=50)
        else:
            _, _, _, A_diff, B_diff, pi_diff, A_diff_init, B_diff_init, pi_diff_init = \
                _hmm.train(_observations, p_tol=1e-6, max_iters=50, hmm_gt=_hmm_gt, dist='l1')
            last_i, last_ll = _hmm.last_i, _hmm.last_ll
            # Plot learning curves
            plot_learning_curves(3, A_diff, B_diff, pi_diff, title='', max_iters=50, last_i=_hmm.last_i,
                                 last_ll=_hmm.last_ll, show=False)
            plot_learning_curves(3, A_diff_init, B_diff_init, pi_diff_init, title='title', max_iters=50,
                                 last_i=_hmm.last_i, last_ll=_hmm.last_ll, secondary=True)

        #   - compare output to ground truth
        #   - (a) Assert Equal to the Ground Truth
        # assert A_opt.__str__(round_places=6) == _hmm_gt.A.__str__(round_places=6), \
        #     f'{A_opt.__str__(round_places=6)} vs {_hmm_gt.A.__str__(round_places=6)}'
        # assert B_opt.__str__(round_places=6) == _hmm_gt.B.__str__(round_places=6), \
        #     f'{B_opt.__str__(round_places=6)} vs {_hmm_gt.B.__str__(round_places=6)}'
        #   - (b) Assert Close to the Ground Truth
        assert close(_hmm.A, _hmm_gt.A, tol=_tol), f'A_opt not close to _hmm_gt.A for tol={_tol}'
        assert close(_hmm.B, _hmm_gt.B, tol=_tol), f'A_opt not close to _hmm_gt.A for tol={_tol}'
        print(f'#{_sample_index:02d} PASSed', file=stderr)
