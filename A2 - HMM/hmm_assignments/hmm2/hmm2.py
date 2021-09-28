import fileinput

from hmm import HMM

if __name__ == '__main__':
    for _sample_index in range(1):
        #   - read files
        _input = fileinput.input(f'hmm2_sample_{_sample_index:02d}.in')
        _output = fileinput.input(f'hmm2_sample_{_sample_index:02d}.ans')
        _output_gt = next(iter(_output)).rstrip()
        #   - initialize HMM
        _hmm, _observations = HMM.from_input(_input)
        _input.close()
        _output.close()
        #   - output most probable states path
        _states_path, _ = _hmm.delta_pass(_observations)
        #   - compare output to ground truth
        assert _states_path.__str__(include_shape=False) == _output_gt, \
            f'_states_path={_states_path.__str__(include_shape=False)} | _output={_output_gt}'
        print('PASSed')
