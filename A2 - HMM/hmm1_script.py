import fileinput
from itertools import chain
from typing import Tuple

from hmm import HMM
from hmm_utils import Matrix2d, Vector


def parse_matrix_2d(data_list : list, shape : list):
    rows = int(shape[0])
    columns = int(shape[1])
    data = [float(x) for x in data_list]
    if rows > 1:
        new_data = [[0.] * columns for _ in range(rows)]
        for i in range(rows):
            for j in range(columns):
                new_data[i][j] = data[j + columns * i]

        return Matrix2d(new_data)
    elif rows == 1:
        return Vector(data)



def main():
    inp = iter(fileinput.input())
    A = Matrix2d.from_str(next(inp))
    B = Matrix2d.from_str(next(inp))
    pi = Vector.from_str(next(inp))
    emissions = [int(x) for x in next(inp).rstrip().split(" ")][1:]

    hmm = HMM(A.shape[0], B.shape[1], A, B, pi)
    ll, alphas = hmm.alpha_pass(emissions)
    print(ll)

if __name__ == '__main__':
    main()