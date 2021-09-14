import math
import random
from sys import stderr
from typing import Optional

from fishing_game_core.game_tree import Node


class MinimaxModelSettings:
    def __init__(self):
        self.max_depth = 5  # up to 5 moves ahead
        self.max_time = 60 * 1e-3  # 60 ms


class MinimaxModel:
    def __init__(self, initial_data: dict, depth: int = 0, settings: Optional[MinimaxModelSettings] = None):
        self.initial_data = initial_data
        self.depth = depth
        self.settings = MinimaxModelSettings() if settings is None else settings

    @staticmethod
    def holding_fish(initial_tree_node: Node) -> bool:
        """
        Check whether our player (aka "we") is currently holding a caught fish.
        :param Node initial_tree_node: root node of the tree
        :return: a bool object
        """
        return initial_tree_node.state.player_caught[0] != -1

    @staticmethod
    def minimax(node: Node, player: int = 0) -> int:
        """
        Minimax algorithm implementation.
        :param Node node: tree node to be expanded (starts from root)
        :param int player: 0 (we) / 1 (opponent)
        :return: an int representation of the branch/move (see MinimaxModel::next_move_minimax())
        """
        raise NotImplementedError

    @staticmethod
    def minimax_with_pruning(node: Node, player: int = 0, alpha: float = -math.inf, beta: float = math.inf) -> int:
        """
        Minimax algorithm implementation with Alpha-Beta Pruning.
        :param Node node: tree node to be expanded (starts from root)
        :param int player: 0 (we) / 1 (opponent)
        :param float alpha: alpha parameter (starts from -Inf)
        :param float beta: beta parameter (starts from +Inf)
        :return: an int representation of the branch/move (see MinimaxModel::next_move_minimax())
        """
        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def next_move_minimax(self, initial_node: Node) -> int:
        """
        Find next best move using the naive Minimax algo, responding in no more than :attr:`self.settings.max_time' in
        seconds, while searching up to :attr:`self.settings.depth` depth.
        :param Node initial_node: root of the new graph to search for the solution
        :return: an int representation of the next best move
                 0: "stay"
                 1: "up"
                 2: "down"
                 3: "left"
                 4: "right"
        """
        # Check if we are currently holding a caught fish
        if MinimaxModel.holding_fish(initial_tree_node=initial_node):
            return 1

        # Compute all children of root node
        children = initial_node.compute_and_get_children()
        print(initial_node.state.player_caught, file=stderr)
        return random.randrange(5)

    # noinspection PyMethodMayBeStatic
    def next_move_minimax_pruning(self, initial_node: Node) -> int:
        """
        Find next best move using the naive Minimax algo + α-β Pruning, responding in no more than
        :attr:`self.settings.max_time' in seconds, while searching up to :attr:`self.settings.depth` depth.
        :param Node initial_node: root of the new graph to search for the solution
        :return: an int representation of the next best move (0: "stay", 1: "up", 2: "down", 3: "left", 4: "right")
        """
        # Check if we are currently holding a caught fish
        if MinimaxModel.holding_fish(initial_tree_node=initial_node):
            return 1
        # TODO
        initial_node.compute_and_get_children()
        return random.randrange(5)
