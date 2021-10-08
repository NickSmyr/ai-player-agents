import fileinput

from hmm import HMM
from hmm_utils import Matrix2d, Vector


def main():
    inp = iter(fileinput.input())
    A = Matrix2d.from_str(next(inp))
    B = Matrix2d.from_str(next(inp))
    pi = Vector.from_str(next(inp))
    emissions = [int(x) for x in next(inp).rstrip().split(" ")][1:]

    hmm = HMM(A.shape[0], B.shape[1], A, B, pi)
    # ll, alphas = hmm.alpha_pass(emissions)
    states, _ = hmm.delta_pass(emissions)
    print(" ".join([str(x) for x in states]))


if __name__ == '__main__':
    main()
