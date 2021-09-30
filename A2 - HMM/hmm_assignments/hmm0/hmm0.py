import fileinput
import os

from hmm import HMM

if __name__ == '__main__':
    print(os.getcwd())
    for _sample_index in range(3):
        #   - read files
        _input = fileinput.input(f'hmm0/hmm0_sample_{_sample_index:02d}.in')
        _output = fileinput.input(f'hmm0/hmm0_sample_{_sample_index:02d}.ans')
        _output_gt = next(iter(_output)).rstrip()
        #   - initialize HMM
        _hmm, _ = HMM.from_input(_input)
        _input.close()
        _output.close()
        #   - output next emission distribution
        _next_emission_dist = _hmm.B_transposed @ (_hmm.A_transposed @ _hmm.pi)
        #   - compare output to ground truth
        _output_str = _next_emission_dist.__str__(round_places=2)
        assert _output_str == _output_gt, f'str(_next_emission_dist)={_output_str} | _output={_output_gt}'
        print('PASSed')
