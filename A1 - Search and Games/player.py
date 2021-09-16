#!/usr/bin/env python3
import math
from sys import stderr
from typing import Tuple

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR

# from minimax import MinimaxModel

MAX_DEPTH = 3
MAX_DEPTH_PRUNING = 3


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
        # MinimaxModel.point_distance_l1(MinimaxModel.INITIAL_HOOK_POSITION, hook_pos) * 1,
        #   - encourage positions in the vicinity of closest fish
        -min([point_distance_l1(hook_pos, fp) * node.state.fish_scores[fi]
              for fi, fp in node.state.fish_positions.items()] if len(node.state.fish_positions) else [0, ]) * 2,
        # 2 * random.random(),
    ])


# noinspection DuplicatedCode
def minimax(node: Node, player: int = 0) -> Tuple[int, float]:
    """
    Minimax algorithm implementation.
    :param Node node: tree node to be expanded (starts from root)
    :param int player: 0 (we) / 1 (opponent)
    :return: an int representation of the branch/move (see MinimaxModel::next_move_minimax())
    """
    children = node.compute_and_get_children()

    # Check if reached leaf nodes or max depth
    if len(children) == 0 or node.depth == MAX_DEPTH:
        return node.move, heuristic(node=node)

    # Recurse
    children_values = [minimax(node=child, player=1 - player)[1] for child in children]
    children_values_len = len(children_values)
    if not node.depth:
        print(f'depth={node.depth}: ' + str(children_values), file=stderr)
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
    else:
        argmin = min(range(len(children_values)), key=lambda v: children_values[v])
        return children[argmin].move, children_values[argmin]


# noinspection DuplicatedCode
def minimax_pruning(node: Node, player: int = 0, alpha: float = -math.inf, beta: float = math.inf) -> Tuple[int, float]:
    """
    Minimax algorithm implementation.
    :param Node node: tree node to be expanded (starts from root)
    :param int player: 0 (we) / 1 (opponent)
    :return: an int representation of the branch/move (see MinimaxModel::next_move_minimax())
    """
    children = node.compute_and_get_children()

    # Check if reached leaf nodes or max depth
    if len(children) == 0 or node.depth == MAX_DEPTH_PRUNING:
        return node.move, heuristic(node=node)

    # Recurse
    if player == 0:
        #   - find values of children
        children_values = []
        for child in children:
            m, v = minimax_pruning(node=child, player=1, alpha=alpha, beta=beta)
            children_values.append(v)
            alpha = max(alpha, v)
            if beta <= alpha:
                break
        children_values_len = len(children_values)
        #   - log
        if not node.depth:
            print(f'depth={node.depth}: ' + str(children_values), file=stderr)
        #   - find max value of children
        argmax = max(range(children_values_len), key=lambda vi: children_values[vi])
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
    else:
        #   - find values of children
        children_values = []
        for child in children:
            m, v = minimax_pruning(node=child, player=0, alpha=alpha, beta=beta)
            children_values.append(v)
            beta = min(beta, v)
            if beta <= alpha:
                break
        #   - find min value of children (rational opponent)
        argmin = min(range(len(children_values)), key=lambda vi: children_values[vi])
        return children[argmin].move, children_values[argmin]


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """
        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


# noinspection PyMethodMayBeStatic
class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """
        # Generate game tree object
        first_msg = self.receiver()
        # Initialize your minimax model
        model = self.initialize_model(initial_data=first_msg)

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(
                model=model, initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def initialize_model(self, initial_data):
        """
        Initialize your minimax model 
        :param initial_data: Game data for initializing minimax model
        :type initial_data: dict
        :return: Minimax model
        :rtype: MinimaxModel

        Sample initial data:
        { 'fish0': {'score': 11, 'type': 3}, 
          'fish1': {'score': 2, 'type': 1}, 
          ...
          'fish5': {'score': -10, 'type': 4},
          'game_over': False }

        Please note that the number of fishes and their types is not fixed between test cases.
        """
        # mm_model = MinimaxModel(initial_data=initial_data, settings=self.settings)
        mm_model = None
        return mm_model

    # noinspection PyMethodMayBeStatic
    def search_best_next_move(self, model, initial_tree_node):
        """
        Use your minimax model to find best possible next move for player 0 (green boat)
        :param model: Minimax model
        :type model: MinimaxModel
        :param initial_tree_node: Initial game tree node 
        :type initial_tree_node: game_tree.Node 
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """
        # Save initial hook position of our boat
        # MinimaxModel.INITIAL_HOOK_POSITION = initial_tree_node.state.hook_positions[0]
        # Compute and return next move using Minimax
        mm_move = minimax(initial_tree_node)[0]
        # mm_move = minimax_pruning(initial_tree_node)[0]
        # mm_move = model.next_move_minimax(initial_node=initial_tree_node)
        # mm_move = model.next_move_minimax_pruning(initial_node=initial_tree_node)
        return ACTION_TO_STR[mm_move]
