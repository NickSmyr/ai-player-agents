import abc
from operator import itemgetter
from typing import Optional, Tuple

from fishing_game_core.game_tree import Node


class MinimaxAgentHParams:
    MAX_DEPTH = 3
    PLAYER_SCORE_MULTIPLIER = 10
    TIMEOUT_DURATION = 60 * 1e-3
    CHECK_REPEATED_STATES = True

    def __init__(self, hparams: dict):
        """
        Initialize a MinimaxAgent hyper-parameters instance.
        :param dict hparams: key-value pairs of hyper-params
        """
        for k, v in hparams.items():
            setattr(self.__class__, k, v)


class MinimaxAgent(abc.ABC):
    # @staticmethod
    @abc.abstractmethod
    def heuristic(self, node: Node) -> float:
        """
        Compute the heuristic function for the given node.
        :param Node node: input tree node or leaf
        :return: a float
        """
        raise NotImplementedError

    @abc.abstractmethod
    def minimax(self, node: Node, player: int, depth: int, alpha: Optional[float] = None,
                beta: Optional[float] = None) -> Tuple[int, float]:
        """
        Perform the Minimax algorithm starting from the given node as root. Optionally prune using α-β pruning or stop
        computations at give depth.
        :param Node node: root
        :param int player: 0 (our boat) / 1 (opponent's boat)
        :param int depth: max depth to stop computations
        :param float alpha: for α-β pruning
        :param float beta: for α-β pruning
        :return: a tuple containing the move (int) and the max value from the heuristic (float)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_next_move(self, initial_node: Node) -> Tuple[int, float]:
        """
        Compute and return best next move using Minimax.
        :param Node initial_node: the root node
        :return: a tuple containing the move (int) and the max value from the heuristic (float)
        """
        raise NotImplementedError


def diff_x(x1, x2) -> Tuple[int, int, int]:
    """
    Compute the differences in x-axis of the two numbers.
    :param x1:
    :param x2:
    :return:
    """
    if x1 > x2:
        min_items = (x1 - x2, 20 + x2 - x1)
        return tuple(min(enumerate(min_items), key=itemgetter(1))) + (0,)
    min_items = (x2 - x1, 20 + x1 - x2)
    return tuple(min(enumerate(min_items), key=itemgetter(1))) + (1,)


def get_node_repr(node: Node) -> str:
    """
    Two nodes have the same representation if the respective state boards seem identical (fishes & hooks at the same
    positions, same player's turn).
    :param Node node: a game tree node
    :return: a node representation as a str object
    """
    state = node.state
    return f'p{state.player}ps{state.player_scores}hp{state.hook_positions}fp{state.fish_positions}'


def holding_fish(initial_tree_node: Node) -> bool:
    """
    Check whether our player (aka "we") is currently holding a caught fish.
    :param Node initial_tree_node: root node of the tree
    :return: a bool object
    """
    return initial_tree_node.state.player_caught[0] != -1


def point_distance_l1(hook_pos: tuple, fish_pos: tuple) -> float:
    """
    Distance between two 2-d points using the Manhattan distance.
    :param tuple hook_pos: (x,y) coordinates of first point
    :param tuple fish_pos: (x,y) coordinates of first point
    :return: distance as a float object
    """
    return min(abs(hook_pos[0] - fish_pos[0]), abs(hook_pos[0] - 20 + fish_pos[0])) + \
        abs(hook_pos[1] - fish_pos[1]) + 20 - fish_pos[1]
