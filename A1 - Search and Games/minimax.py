from game_tree import Node


class MinimaxModelSettings:
    def __init__(self):
        self.max_depth = 5  # up to 5 moves ahead
        self.max_time = 60 * 1e-3  # 60 ms


class MinimaxModel:
    def __init__(self, initial_data: dict, depth: int = 0):
        self.initial_data = initial_data
        self.depth = depth
        self.settings = MinimaxModelSettings()

    # noinspection PyMethodMayBeStatic
    def next_move_minimax(self, initial_node: Node) -> str:
        """
        Find next best move using the naive minimax algo, responding in no more than :attr:`self.settings.max_time' in
        seconds, while searching up to :attr:`self.settings.depth` depth.
        :param Node initial_node: root of the new graph to search for the solution
        :return: a string representation of the next best move ('left', 'right', 'up', 'down', 'stay')
        """
        # TODO
        return 'right'
