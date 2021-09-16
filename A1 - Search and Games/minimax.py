import math
import random
from typing import Optional, Tuple, List

from fishing_game_core.game_tree import Node
from main import Settings


class MinimaxModel:
    MAX_DEPTH = 3
    MAX_DEPTH_PRUNING = 5

    INITIAL_HOOK_POSITION = None

    def __init__(self, initial_data: dict, depth: int = 0, settings: Optional[Settings] = None):
        self.initial_data = initial_data
        self.depth = depth
        self.settings = settings

    @staticmethod
    def holding_fish(initial_tree_node: Node) -> bool:
        """
        Check whether our player (aka "we") is currently holding a caught fish.
        :param Node initial_tree_node: root node of the tree
        :return: a bool object
        """
        return initial_tree_node.state.player_caught[0] != -1

    @staticmethod
    def point_distance_l1(point1: tuple, point2: tuple, obstacle: tuple = None) -> float:
        """
        Distance between two 2-d points using the Manhattan distance.
        :param tuple point1: (x,y) coordinates of first point
        :param tuple point2: (x,y) coordinates of first point
        :param tuple obstacle: (x,y) coordinates of obstacle point (i.e. the opponent's boat)
        :return: distance as a float object
        """
        dist = abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
        # print(point1, point2, dist)
        return dist

    @staticmethod
    def point_distance_l2(point1: tuple, point2: tuple, obstacle: tuple = None) -> float:
        """
        Distance between two 2-d points using the Euclidean distance.
        :param tuple point1: (x,y) coordinates of first point
        :param tuple point2: (x,y) coordinates of first point
        :param tuple obstacle: (x,y) coordinates of obstacle point (i.e. the opponent's boat)
        :return: distance as a float object
        """
        return (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2

    @staticmethod
    def heuristic(node: Node) -> float:
        """
        Heuristic function to compute the value at any given node/state.
        :param Node node: a graph node as game_tree.Node object
        :return: the heuristic value as a float
        """
        # return 2 * (node.state.player_scores[0] - node.state.player_scores[1]) + \
        #        100 * MinimaxModel.point_distance_l1(node.state.hook_positions[0], node.state.hook_positions[1])
        hook_pos = node.state.hook_positions[0]
        return sum([
            #   - encourage winning
            (node.state.player_scores[0] - node.state.player_scores[1]) * 10,
            #   - discourage collisions
            MinimaxModel.point_distance_l1(MinimaxModel.INITIAL_HOOK_POSITION, hook_pos) * 1,
            #   - encourage positions in the vicinity of closest fish
            -min([MinimaxModel.point_distance_l1(hook_pos, fp) * node.state.fish_scores[fi]
                  for fi, fp in node.state.fish_positions.items()] if len(node.state.fish_positions) else [0, ]) * 10,
            2 * random.random(),
        ])
        # + 100 * min(
        # [MinimaxModel.point_distance_l2(node.state.hook_positions[0], fp, node.state.hook_positions[1])
        #  for _, fp in node.state.fish_positions.items()])

    @staticmethod
    def minimax(node: Node, player: int = 0) -> Tuple[int, float]:
        """
        Minimax algorithm implementation.
        :param Node node: tree node to be expanded (starts from root)
        :param int player: 0 (we) / 1 (opponent)
        :return: an int representation of the branch/move (see MinimaxModel::next_move_minimax())
        """
        children: List[Node] = node.compute_and_get_children()
        # Check if reached leaf nodes or max depth
        if len(children) == 0 or node.depth == MinimaxModel.MAX_DEPTH:
            return node.move, MinimaxModel.heuristic(node=node)
        # Recurse
        children_values = [MinimaxModel.minimax(node=child, player=1 - player)[1] for child in children]
        children_values_len = len(children_values)
        # if not depth:
        #     print(f'depth={depth}: ' + str(children_values))
        if player == 0:
            #   - find max value of children
            argmax = max(range(children_values_len), key=lambda v: children_values[v])
            max_value = children_values[argmax]
            #   - in case of equal value, select the move <> 0
            if node.depth == 0:
                for i in range(argmax, children_values_len):
                    if children_values[i] == max_value and children[i].move != 0:
                        return children[i].move, max_value
            return children[argmax].move, max_value
        #     v_max = -math.inf
        #     node_max = None
        #     for child in children:
        #         v = MinimaxModel.minimax(node=child, player=1)[1]
        #         if v > v_max:
        #             v_max = v
        #             node_max = child
        #     if node.depth == 0:
        #         print(f'v={v}')
        #     return node_max.move, v_max
        # # Min
        # else:
        #     v_min = +math.inf
        #     node_min = None
        #     for child in children:
        #         v = MinimaxModel.minimax(node=child, player=0)[1]
        #         if v < v_min:
        #             v_min = v
        #             node_min = child
        #     return node_min.move, v_min
        argmin = min(range(len(children_values)), key=lambda v: children_values[v])
        return children[argmin].move, children_values[argmin]

    @staticmethod
    def minimax_with_pruning(node: Node, player: int = 0, alpha: float = -math.inf,
                             beta: float = math.inf) -> Tuple[int, float]:
        """
        Minimax algorithm implementation with Alpha-Beta Pruning.
        :param Node node: tree node to be expanded (starts from root)
        :param int player: 0 (we) / 1 (opponent)
        :param float alpha: alpha parameter (starts from -Inf)
        :param float beta: beta parameter (starts from +Inf)
        :return: an int representation of the branch/move (see MinimaxModel::next_move_minimax())
        """
        children: List[Node] = node.compute_and_get_children()
        # Check if reached leaf nodes or max depth
        if len(children) == 0 or node.depth == MinimaxModel.MAX_DEPTH_PRUNING:
            return node.move, MinimaxModel.heuristic(node=node)
        # Recurse
        move = None
        if player == 0:
            max_value = -math.inf
            for child in children:
                move, value = MinimaxModel.minimax_with_pruning(node=child, player=1, alpha=alpha, beta=beta)
                max_value = max(max_value, value)
                alpha = max(max_value, alpha)
                if beta <= alpha:
                    break
            if node.depth == 0:
                print(f'depth={node.depth}: ' + str((move, max_value)))
            return move, max_value
        # Opponent
        else:
            min_value = math.inf
            for child in children:
                move, value = MinimaxModel.minimax_with_pruning(node=child, player=0, alpha=alpha, beta=beta)
                min_value = min(min_value, value)
                beta = min(min_value, beta)
                if beta <= alpha:
                    break
            return move, min_value

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
        # Compute next best move via Minimax
        return MinimaxModel.minimax(node=initial_node)[0]

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
        # Compute next best move via Minimax
        return MinimaxModel.minimax_with_pruning(node=initial_node)[0]
