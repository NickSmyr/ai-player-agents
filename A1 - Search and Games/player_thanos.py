#!/usr/bin/env python3
import math
import random
import time
from operator import itemgetter
from sys import stderr
from typing import Tuple

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR
# from minimax.naive import NaiveAgent
from minimax_thanos.ids import IDSAgent
from minimax_thanos.utils import MinimaxAgentHParams

MAX_DEPTH = 3
MAX_DEPTH_PRUNING = 7
MAX_DEPTH_IDS = 99999
RS_COUNT = 0
EXPLORED_SET = {}
IDS_VALUES = {}  # serial saving
INITIAL_NODE_REPR = None
TIME_START = 0.
TIME_THRESHOLD = 50 * 1e-3

MOVES_STILL = 0
MOVES_STILL_THRESHOLD = 10

TOTAL_FISH_SCORE = 0  # sum of the positive scores
TOTAL_FISH_SCORE_HALF = 0  # half of sum of the positive scores
TOTAL_FISH_SCORE_WN = 0  # sum with negative scores

PLAYER_SCORE_MULTIPLIER = 10
FISH_SCORE_MULTIPLIER = 5
FISH_DIST_MULTIPLIER = 2

OPPOSITE_OF_PRE_PREVIOUS_MOVE = 0
OPPOSITE_OF_PREVIOUS_MOVE = 0
OPPOSITE_OF_MOVE = {
    0: 0,
    1: 2,
    2: 1,
    3: 4,
    4: 3
}


def point_distance_l1(point1: tuple, point2: tuple, obstacle: tuple = None, score: int = 0) -> float:
    """
    Distance between two 2-d points using the Manhattan distance.
    :param tuple point1: (x,y) coordinates of first point
    :param tuple point2: (x,y) coordinates of first point
    :param tuple obstacle: (x,y) coordinates of obstacle point (i.e. the opponent's boat)
    :param int score: fish score
    :return: distance as a float object
    """
    dist = abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
    # print(point1, point2, dist)
    return dist - score


def heuristic(node: Node) -> float:
    """
    Heuristic function to compute the value at any given node/state.
    :param Node node: a graph node as game_tree.Node object
    :return: the heuristic value as a float
    """
    our_score = node.state.player_scores[0]
    opp_score = node.state.player_scores[1]
    fishes_count = len(node.state.fish_positions)

    # Check if state is a terminating one
    if fishes_count == 0:
        return (our_score - opp_score) * PLAYER_SCORE_MULTIPLIER * 10

    # Check if state is a losing one
    if opp_score > TOTAL_FISH_SCORE_HALF:
        return -math.inf
    # Check if state is a winning one
    if our_score > TOTAL_FISH_SCORE_HALF:
        return +math.inf
    # Check if there is only one fish left
    our_hook_pos, oop_hook_pos = node.state.hook_positions.values()
    # if fishes_count == 1:
    #     last_fish_score = node.state.fish_scores[list(node.state.fish_positions.keys())[0]]
    #     return (abs(last_fish_score) // last_fish_score) * PLAYER_SCORE_MULTIPLIER
    #     if max(opp_score, opp_score + last_fish_score) > TOTAL_FISH_SCORE_HALF:
    #         return -math.inf
    #     # Check if state is a winning one
    #     if (our_score + last_fish_score) > TOTAL_FISH_SCORE_HALF:
    #         return +math.inf
    #     return (our_score + last_fish_score - opp_score) * PLAYER_SCORE_MULTIPLIER
    fv = 0
    for fi, fp in node.state.fish_positions.items():
        fd = point_distance_l1(our_hook_pos, fp, oop_hook_pos)
        fs = node.state.fish_scores[fi]
        if fd == 0 and fs > 0:
            return math.inf if fishes_count == 1 else 100 * FISH_SCORE_MULTIPLIER
        # if fd <= 2 and our_score + fs > TOTAL_FISH_SCORE_HALF:
        #     return (abs(fs) // fs) * PLAYER_SCORE_MULTIPLIER + 1
        fv += FISH_SCORE_MULTIPLIER * fs / fd if fd != 0 \
            else FISH_SCORE_MULTIPLIER * node.state.fish_scores[fi]
    # return 2 * (node.state.player_scores[0] - node.state.player_scores[1]) + \
    #        100 * MinimaxModel.point_distance_l1(node.state.hook_positions[0], node.state.hook_positions[1])
    return (our_score - opp_score) * PLAYER_SCORE_MULTIPLIER + fv / fishes_count
    # return sum([
    #     #   - encourage winning
    #     (our_score - opp_score) * PLAYER_SCORE_MULTIPLIER,
    #     #   - discourage collisions
    #     # point_distance_l1(MinimaxModel.INITIAL_HOOK_POSITION, hook_pos) * 1,
    #     #   - encourage positions in the vicinity of closest fish
    #     -min([point_distance_l1(hook_pos, fp, score=FISH_SCORE_MULTIPLIER * node.state.fish_scores[fi])
    #           for fi, fp in node.state.fish_positions.items()] if len(node.state.fish_positions) else [0, ]) * 1,
    #     # 2 * random.random(),
    # ])


# def heuristic(node):
#     state = node.state
#     scores = state.player_scores
#     fish_positions = state.fish_positions
#     len_fish_positions = len(fish_positions)
#
#     if len_fish_positions == 0:
#         return 100 * (scores[0] - scores[1])
#
#     our_position, their_position = state.get_hook_positions().values()
#     v = 0
#     for fish, position in fish_positions.items():
#         dist = point_distance_l1(position, our_position, their_position)
#         if dist == 0 and len_fish_positions == 1:
#             return math.inf
#         v += 5 * state.fish_scores[fish] / dist if dist != 0 else 5 * state.fish_scores[fish]
#
#     v = 10 * (scores[0] - scores[1]) + (v / len_fish_positions) - len_fish_positions
#     return v

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
    # if not node.depth:
    #     print(f'depth={node.depth}: ' + str(children_values), file=stderr)
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
    else:
        argmin = min(range(len(children_values)), key=lambda v: children_values[v])
        return children[argmin].move, children_values[argmin]


def get_node_repr(node: Node) -> str:
    """
    Two nodes have the same representation if the respective state boards seem identical (fishes & hooks at the same
    positions, same player's turn).
    :param Node node: a game tree node
    :return: a node representation as a str object
    """
    return f'p{node.state.player}hp{node.state.hook_positions}fp{node.state.fish_positions}'


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


# noinspection DuplicatedCode
def minimax_pruning_max(node: Node, alpha: float = -math.inf, beta: float = math.inf,
                        max_depth: int = MAX_DEPTH_PRUNING) -> Tuple[int, float]:
    """
    Minimax algorithm implementation for MAX player (our boat).
    :param Node node: tree node to be expanded (starts from root)
    :param float alpha: alpha parameter of Alpha-Beta Pruning algorithm
    :param float beta: beta parameter of Alpha-Beta Pruning algorithm
    :param float max_depth: maximum search length in moves
    :return: an int representation of the branch/move (see MinimaxModel::next_move_minimax())
    """
    global EXPLORED_SET
    # Repeated states checking
    node_repr = get_node_repr(node=node)
    if node_repr in EXPLORED_SET:
        return EXPLORED_SET[node_repr].values()

    if time.time() - TIME_START > TIME_THRESHOLD:
        raise TimeoutError

    # if node_repr == INITIAL_NODE_REPR:
    #     return 0, +1000.

    # Get all children nodes of current
    children = sorted(node.compute_and_get_children(), key=heuristic, reverse=True)

    # Check if reached leaf nodes or max depth
    if len(children) == 0 or max_depth == 0:
        return node.move, heuristic(node=node)

    # # Remove "stay" move if still for too many moves
    # if node.depth == 0 and MOVES_STILL >= MOVES_STILL_THRESHOLD:
    #     to_remove = None
    #     child: Node
    #     for i, child in enumerate(children):
    #         if child.move == 0:
    #             to_remove = i
    #             break
    #     if to_remove is not None:
    #         del children[to_remove]

    # # Remove "up" move if not valid
    # our_hook_pos = node.state.hook_positions[0]
    # opp_hook_pos = node.state.hook_positions[1]
    # if our_hook_pos[1] == 19 and node.state.player_caught[0] == -1:
    #     to_remove = None
    #     child: Node
    #     for i, child in enumerate(children):
    #         if child.move == 1:
    #             to_remove = i
    #             break
    #     if to_remove is not None:
    #         del children[to_remove]
    #
    # # Remove left/right move
    # hook_dist_x, dist_min_idx, dist_right_idx = diff_x(our_hook_pos[0], opp_hook_pos[0])
    # if 1 == hook_dist_x:
    #     to_remove = None
    #     child: Node
    #     for i, child in enumerate(children):
    #         if child.move == 4 and (dist_min_idx + dist_right_idx == 1) or \
    #                 child.move == 3 and (dist_min_idx * dist_right_idx == 1 or dist_min_idx + dist_right_idx == 0):
    #             to_remove = i
    #             break
    #     if to_remove is not None:
    #         del children[to_remove]

    # Recurse
    #   - find values of children
    argmax = 0
    args_max = []
    max_value = -math.inf
    for i, child in enumerate(children):
        m, v = minimax_pruning_min(node=child, alpha=alpha, beta=beta, max_depth=max_depth - 1)
        # children_values.append(v)
        if v > max_value:
            max_value = v
            argmax = i
            args_max = [i, ]
        elif v == max_value:
            args_max.append(i)
        alpha = max(alpha, max_value)
        if beta <= alpha:
            break
    #   - in case of equal value, select the move <> 0
    if node.depth == 0 and len(args_max) > 1:
        # print(f'depth={node.depth}: ' + str(max_value), file=stderr)
        for am in args_max:
            if children[am].move != OPPOSITE_OF_PREVIOUS_MOVE:
                return children[am].move, max_value
    # Store node in the explored set (Graph Version)
    EXPLORED_SET[node_repr] = {'move': children[argmax].move, 'value': max_value}
    # IDS_VALUES[node_repr] = max_value
    return children[argmax].move, max_value


# noinspection DuplicatedCode
def minimax_pruning_min(node: Node, alpha: float = -math.inf, beta: float = math.inf,
                        max_depth: int = MAX_DEPTH_PRUNING) -> Tuple[int, float]:
    """
    Minimax algorithm implementation for MIN player (opponent's boat).
    :param Node node: tree node to be expanded (starts from root)
    :param float alpha: alpha parameter of Alpha-Beta Pruning algorithm
    :param float beta: beta parameter of Alpha-Beta Pruning algorithm
    :param float max_depth: maximum search length in moves
    :return: an int representation of the branch/move (see MinimaxModel::next_move_minimax())
    """
    if time.time() - TIME_START > TIME_THRESHOLD:
        raise TimeoutError

    global EXPLORED_SET
    # Repeated states checking
    node_repr = get_node_repr(node=node)
    if node_repr in EXPLORED_SET:
        return EXPLORED_SET[node_repr].values()

    # Get all children nodes of current
    children = sorted(node.compute_and_get_children(), key=heuristic, reverse=True)

    # Check if reached leaf nodes or max depth
    if len(children) == 0 or node.depth == MAX_DEPTH_PRUNING:
        return node.move, heuristic(node=node)

    # # Remove left/right move
    # our_hook_pos = node.state.hook_positions[0]
    # opp_hook_pos = node.state.hook_positions[1]
    # hook_dist_x, dist_min_idx, dist_right_idx = diff_x(our_hook_pos[0], opp_hook_pos[0])
    # if 1 == hook_dist_x:
    #     to_remove = None
    #     child: Node
    #     for i, child in enumerate(children):
    #         if child.move == 3 and (dist_min_idx + dist_right_idx == 1) or \
    #                 child.move == 4 and (dist_min_idx * dist_right_idx == 1 or dist_min_idx + dist_right_idx == 0):
    #             to_remove = i
    #             break
    #     if to_remove is not None:
    #         children.pop(to_remove)

    # Recurse
    # - find values of children
    argmin = 0
    min_value = math.inf
    for i, child in enumerate(children):
        m, v = minimax_pruning_max(node=child, alpha=alpha, beta=beta, max_depth=max_depth - 1)
        #   - find min value of children (rational opponent)
        if v < min_value:
            min_value = v
            argmin = i
        beta = min(beta, min_value)
        if beta <= alpha:
            break
    # Store node in the explored set (Graph Version)
    EXPLORED_SET[node_repr] = {'move': children[argmin].move, 'value': min_value}
    # IDS_VALUES[node_repr] = min_value
    return children[argmin].move, min_value


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
        model = self.initialize_model2(initial_data=first_msg)

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move2(
                model=model, initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def initialize_model2(self, initial_data):
        # return NaiveAgent(initial_data, hparams=HParams({
        #     'MAX_DEPTH': 3,
        # }))
        # return ABAgent(initial_data, hparams=HParams({
        #     'MAX_DEPTH': 6,
        # }))
        return IDSAgent(initial_data, hparams=MinimaxAgentHParams({
            'MAX_DEPTH': MAX_DEPTH_IDS,
            'TIMEOUT_DURATION': TIME_THRESHOLD,
        }))

    def search_best_next_move2(self, model, initial_tree_node):
        return ACTION_TO_STR[model.get_next_move(initial_tree_node)[0]]

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
        global TOTAL_FISH_SCORE, TOTAL_FISH_SCORE_WN, TOTAL_FISH_SCORE_HALF
        # mm_model = MinimaxModel(initial_data=initial_data, settings=self.settings)
        k: str
        v: dict
        for k, v in initial_data.items():
            if k.startswith('fish'):
                fish_score = v['score']
                TOTAL_FISH_SCORE += fish_score if fish_score > 0 else 0
                TOTAL_FISH_SCORE_WN += fish_score
        TOTAL_FISH_SCORE_HALF = TOTAL_FISH_SCORE // 2
        return None

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
        global EXPLORED_SET, INITIAL_NODE_REPR, OPPOSITE_OF_PREVIOUS_MOVE, TIME_START, MOVES_STILL
        # Save initial hook position of our boat
        # MinimaxModel.INITIAL_HOOK_POSITION = initial_tree_node.state.hook_positions[0]
        # Compute and return next move using Minimax
        # mm_move, _ = minimax(initial_tree_node)
        INITIAL_NODE_REPR = get_node_repr(initial_tree_node)
        TIME_START = time.time()

        final_mm_move = 0
        old_mm_value = -math.inf
        for d in range(MAX_DEPTH_PRUNING):
            EXPLORED_SET = {}
            try:
                mm_move, mm_value = minimax_pruning_max(initial_tree_node, max_depth=d)
                if mm_value > old_mm_value:
                    final_mm_move = mm_move
                    old_mm_value = mm_value
            except TimeoutError:
                break

        # mm_move = model.next_move_minimax(initial_node=initial_tree_node)
        # mm_move = model.next_move_minimax_pruning(initial_node=initial_tree_node)
        # OPPOSITE_OF_PRE_PREVIOUS_MOVE = OPPOSITE_OF_PREVIOUS_MOVE
        if final_mm_move is None:
            print('\t> None --> 0', file=stderr)
            final_mm_move = 0

        # Force it to move!
        if final_mm_move == 0 and MOVES_STILL > MOVES_STILL_THRESHOLD:
            print('\t>>> Forcing it to move...', file=stderr)
            return ACTION_TO_STR[random.randint(2, 4)]

        if final_mm_move in (3, 4):
            MOVES_STILL = 0
        else:
            MOVES_STILL += 1
        OPPOSITE_OF_PREVIOUS_MOVE = OPPOSITE_OF_MOVE[final_mm_move]
        print(f'---> MOVE = {ACTION_TO_STR[final_mm_move]}', file=stderr)
        self.sender({"action": final_mm_move, "search_time": None})
        return ACTION_TO_STR[final_mm_move]
