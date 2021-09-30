import fileinput

from hmm import HMM
from hmm_utils import Matrix2d, Vector


_hmm = HMM.from_input(fileinput.input())
print(_hmm.A)
print(_hmm.B)
print(_hmm.pi)
