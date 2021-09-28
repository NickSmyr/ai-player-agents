import fileinput

from hmm import HMM
from hmm_utils import Matrix2d, Vector


def parse_matrix_2d(data_list: list, shape: list):
    rows = shape[0]
    columns = shape[1]
    data = [float(x) for x in data_list]
    if rows[0] > 1:
        new_data = [[0.] * columns] * rows
        for i in range(rows):
            for j in range(columns):
                new_data = data[j + columns * j]
        return Matrix2d(new_data)
    elif rows[0] == 1:
        return Vector(data)


#
# inp = iter(fileinput.input())
# transition_matrix = [int(x) for x in next(inp).rstrip().split(" ")]
# transition_matrix = parse_matrix_2d(transition_matrix[2:], transition_matrix[:2])
# emission_matrix = [int(x) for x in next(inp).rstrip().split(" ")]
# emission_matrix = parse_matrix_2d(emission_matrix[2:], emission_matrix[:2])
# initial_p = [int(x) for x in next(inp).rstrip().split(" ")]
# initial_p = parse_matrix_2d(initial_p[2:], initial_p[:2])
#
# # emission_matrix (N, K)
# output = (transition_matrix @ initial_p)
_hmm, _ = HMM.from_input(fileinput.input())
print(_hmm.A)
print(_hmm.B)
print(_hmm.pi)
