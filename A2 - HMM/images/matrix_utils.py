import random


class Vector(list):
    def __init__(self, data: list):
        self.data = data
        self.n = len(self.data)
        assert type(data[0]), f'Input not a vector (type of data[0]={type(data[0])})'
        list.__init__(self, data)

    def normalize(self) -> None:
        s = sum(self.data)
        self.data = [self.data[i] / s for i in range(self.n)]

    def sum(self):
        return sum(self.data)

    def __matmul__(self, B):
        # TODO
        pass

    @staticmethod
    def random(n: int) -> 'Vector':
        return Vector([random.random() for _ in range(n)])


class Matrix2d(list):
    def __init__(self, data: list):
        self.data = data
        assert len(data[1]) == len(data[0]), f'Dims not match len(data[0])={len(data[0])}, len(data[1])={len(data[1])}'
        self.nrows = len(self.data)
        self.ncols = len(self.data[0])
        list.__init__(self, data)

    def sum_row(self, r: int):
        return sum(self.data[r])

    def sum_rows(self):
        sums = [0.] * self.nrows
        for r in range(self.nrows):
            sums[r] = self.sum_row(r)
        return sums

    def normalize_rows(self) -> None:
        for r in range(self.nrows):
            row_sum = sum(self.data[r])
            self.data[r] = [self.data[r][i] / row_sum for i in range(self.ncols)]

    def T(self):
        self.data = zip(*self.data)

    def get_col(self, c: int) -> Vector:
        vdata = [0.] * self.nrows
        for r in range(self.nrows):
            vdata[r] = self.data[r][c]
        return Vector(vdata)

    def get_row(self, r: int) -> Vector:
        vdata = [0.] * self.ncols
        for c in range(self.ncols):
            vdata[c] = self.data[r][c]
        return Vector(vdata)

    @staticmethod
    def random(nrows: int, ncols: int) -> 'Matrix2d':
        return Matrix2d([[random.random() for _ in range(ncols)] for _ in range(nrows)])
