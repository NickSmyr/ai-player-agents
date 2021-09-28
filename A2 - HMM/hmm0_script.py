import fileinput

from hmm import HMM

_hmm = HMM.from_input(fileinput.input())
print(_hmm.A)
print(_hmm.B)
print(_hmm.pi)
