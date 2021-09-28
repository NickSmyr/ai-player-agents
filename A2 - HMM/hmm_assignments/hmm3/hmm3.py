import fileinput

from hmm import HMM

if __name__ == '__main__':
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
        A_opt, B_opt, _ = _hmm.baum_welch(_observations, tol=1e-10, max_iter=100)
        #   - compare output to ground truth
        assert A_opt.__str__(round_places=6) == _hmm_gt.A.__str__(round_places=6), \
            f'{A_opt.__str__(round_places=6)} vs {_hmm_gt.A.__str__(round_places=6)}'
        assert B_opt.__str__(round_places=6) == _hmm_gt.B.__str__(round_places=6), \
            f'{B_opt.__str__(round_places=6)} vs {_hmm_gt.B.__str__(round_places=6)}'
        print('PASSed')
