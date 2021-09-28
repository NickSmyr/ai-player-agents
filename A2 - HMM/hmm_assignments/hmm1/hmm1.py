import fileinput

from hmm import HMM

if __name__ == '__main__':
    for _sample_index in range(1):
        #   - read files
        _input = fileinput.input(f'hmm1_sample_{_sample_index:02d}.in')
        _output = fileinput.input(f'hmm1_sample_{_sample_index:02d}.ans')
        _output_gt = next(iter(_output)).rstrip()
        #   - initialize HMM
        _hmm, _observations = HMM.from_input(_input)
        _input.close()
        _output.close()
        #   - output next emission distribution
        _obs_likelihood, _, _ = _hmm.alpha_pass(_observations)
        #   - compare output to ground truth
        assert f'{_obs_likelihood:.6f}' == _output_gt, f'_obs_likelihood={_obs_likelihood:.6f} | _output={_output_gt}'
        print('PASSed')
