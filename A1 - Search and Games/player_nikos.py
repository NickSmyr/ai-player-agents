#!/usr/bin/env python3
import math
import sys
import time
from sys import stderr
from typing import Tuple

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR

# from minimax import MinimaxModel
from hparams import MAX_DEPTH, MAX_DEPTH_PRUNING, TIME_LIMIT


def point_distance_l1(point1: tuple, point2: tuple, obstacle: tuple = None) -> float:
    """
    Distance between two 2-d points using the Manhattan distance.
    Each coordinate item can take values from 0 to 19

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
    value = sum([
           #- encourage winning
        (node.state.player_scores[0] - node.state.player_scores[1]) * 10,
           #- discourage collisions
         #point_distance_l1(MinimaxModel.INITIAL_HOOK_POSITION, hook_pos) * 1,
           #- encourage positions in the vicinity of closest fish
        -min([point_distance_l1(hook_pos, fp) * node.state.fish_scores[fi]
              for fi, fp in node.state.fish_positions.items()] if len(node.state.fish_positions) else [0, ]) * 2,
         #2 * random.random(),
    ])
    node.heuristic = value
    return value
    #return node.state.player_scores[0] - node.state.player_scores[1]


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


def get_node_repr(node: Node) -> str:
    """
    Two nodes have the same representation if the respective state boards seem identical (fishes & hooks at the same
    positions, same player's turn).
    :param Node node: a game tree node
    :return: a node representation as a str object
    """
    return f'p{node.state.player}hp{node.state.hook_positions}fp{node.state.fish_positions}'


class Timer():
    """
    Timer class. Creates a timer with a time limit on startup.
    """
    def __init__(self, time_limit):
        self.time_limit = time_limit
        self.initial_time = time.time()

    def timesup(self) -> bool:
        """
        Returns true if the timer has expired
        """
        #print("TIMESUP " , file=stderr)
        return (time.time() - self.initial_time) > self.time_limit

class IDS:
    def __init__(self, time_limit):
        """
        Creates the IDS, model. The solution is returned after at most time_limit seconds
        """
        self.time_limit= time_limit

    def find_solution(self, node: Node):
        """
        Find the best solution within the time span
        """
        timer = Timer(self.time_limit)
        best_move = None
        best_value = -math.inf
        for solution in self.iterate_solutions(node, timer):
            move, value = solution
            best_value = value
            best_move = move

        if best_move is not None:
            return best_move, best_value
        else:
            return 0, -1

    def iterate_solutions(self, node: Node, timer : Timer):
        """
        Iterate over solutions
        """
        depth = 1
        while True:
            if depth == 999999:
                return
            try:
                # Find solutions of increasing depth
                #print(f"Minimax for depth {depth}")
                #print(f"Starting search for depth {depth}", file=stderr)
                self.mm_model = IDSMinimaxPruner()
                move, value = self.mm_model.minimax_pruning(node, depth=depth, timer=timer)
                #print(f"Output move {ACTION_TO_STR[move]} and value {value}")
                yield move, value
                depth += 1
            # If stop iteration raised by mm model then stop
            except StopIteration:
                print("Maximum depth reached ", depth, file=sys.stderr)
                return


class IDSMinimaxPruner:
    def __init__(self):
        self.explored_set = {}

    def minimax_pruning(self, node: Node, player: int = 0, alpha: float = -math.inf, beta: float = math.inf,
                        depth: int = None, timer : Timer = None) -> Tuple[int, float]:
        """
        Minimax algorithm implementation.
        :param Node node: tree node to be expanded (starts from root)
        :param int player: 0 (we) / 1 (opponent)
        :return: an int representation of the branch/move (see MinimaxModel::next_move_minimax())
        """
        #print(f"Start of recursion with depth {depth} ", file=stderr)
        # Check if we have run out of time
        if timer is not None and timer.timesup():
            #print("recursion ended due to time " , file=stderr)
            raise StopIteration
        # Repeated states checking
        node_repr = get_node_repr(node=node)
        if node_repr in self.explored_set:
            #print("recursion ended due to explored " , file=stderr)
            return self.explored_set[node_repr].values()

        # Get all children nodes of current
        children = node.compute_and_get_children()

        # Check if reached leaf nodes or max depth
        if len(children) == 0 or node.depth == depth:
            #print("recursion ended due to terminal node or max depth " , file=stderr)
            return node.move if node.move is not None else 0, heuristic(node=node)

        # Recurse
        if player == 0:
            #   - find values of children
            # children_values = []
            def appraiser(x : Node):
                return x.heuristic if "heuristic" in x.__dict__ else -math.inf
            children = sorted(children, key=appraiser, reverse=True)
            argmax = 0
            args_max = []
            max_value = -math.inf
            for i, child in enumerate(children):
                m, v = self.minimax_pruning(node=child, player=1, alpha=alpha, beta=beta, depth=depth, timer=timer)
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
                #print(f'depth={node.depth}: ' + str(max_value), file=stderr)
                for am in args_max:
                    if children[am].move != 0:
                        return children[am].move, max_value
            # Store node in the explored set (Graph Version)
            self.explored_set[get_node_repr(node=node)] = {'move': children[argmax].move, 'value': max_value}
            return children[argmax].move, max_value

        else:
            #   - find values of children
            def appraiser(x : Node):
                return x.heuristic if "heuristic" in x.__dict__ else math.inf

            children = sorted(children, key=appraiser)
            argmin = 0
            min_value = math.inf
            for i, child in enumerate(children):
                m, v = self.minimax_pruning(node=child, player=0, alpha=alpha, beta=beta, depth=depth, timer=timer)
                #   - find min value of children (rational opponent)
                if v < min_value:
                    min_value = v
                    argmin = i
                beta = min(beta, min_value)
                if beta <= alpha:
                    break
            # Store node in the explored set (Graph Version)
            self.explored_set[get_node_repr(node=node)] = {'move': children[argmin].move, 'value': min_value}
            return children[argmin].move, min_value


# noinspection DuplicatedCode
class MinimaxPruner:
    def __init__(self):
        self.explored_set = {}

    def minimax_pruning(self, node: Node, player: int = 0, alpha: float = -math.inf, beta: float = math.inf) -> Tuple[
        int, float]:
        """
        Minimax algorithm implementation.
        :param Node node: tree node to be expanded (starts from root)
        :param int player: 0 (we) / 1 (opponent)
        :return: an int representation of the branch/move (see MinimaxModel::next_move_minimax())
        """
        # Repeated states checking
        node_repr = get_node_repr(node=node)
        if node_repr in self.explored_set:
            return self.explored_set[node_repr].values()

        # Get all children nodes of current
        children = node.compute_and_get_children()

        # Check if reached leaf nodes or max depth
        if len(children) == 0 or node.depth == MAX_DEPTH_PRUNING:
            return node.move, heuristic(node=node)

        # Recurse
        if player == 0:
            #   - find values of children
            # children_values = []
            argmax = 0
            args_max = []
            max_value = -math.inf
            for i, child in enumerate(children):
                m, v = self.minimax_pruning(node=child, player=1, alpha=alpha, beta=beta)
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
                print(f'depth={node.depth}: ' + str(max_value), file=stderr)
                for am in args_max:
                    if children[am].move != 0:
                        return children[am].move, max_value
            # Store node in the explored set (Graph Version)
            self.explored_set[get_node_repr(node=node)] = {'move': children[argmax].move, 'value': max_value}
            return children[argmax].move, max_value

        else:
            #   - find values of children
            argmin = 0
            min_value = math.inf
            for i, child in enumerate(children):
                m, v = self.minimax_pruning(node=child, player=0, alpha=alpha, beta=beta)
                #   - find min value of children (rational opponent)
                if v < min_value:
                    min_value = v
                    argmin = i
                beta = min(beta, min_value)
                if beta <= alpha:
                    break
            # Store node in the explored set (Graph Version)
            self.explored_set[get_node_repr(node=node)] = {'move': children[argmin].move, 'value': min_value}
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
        #self.mm_model = MinimaxPruner()
        self.mm_model = IDS(TIME_LIMIT) # 75 msec

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
        # mm_move, _ = minimax(initial_tree_node)
        mm_move, val = self.mm_model.find_solution(initial_tree_node)
        # mm_move = model.next_move_minimax(initial_node=initial_tree_node)
        # mm_move = model.next_move_minimax_pruning(initial_node=initial_tree_node)
        print(f"Move {ACTION_TO_STR[mm_move]} with val {val}", file=sys.stderr)
        return ACTION_TO_STR[mm_move]
