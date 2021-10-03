import fileinput
from sys import stderr

import numpy as np

from hmm import HMM


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
        _hmm.baum_welch(_observations, tol=1e-6, max_iter=50, A_gt=_hmm_gt.A)
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
