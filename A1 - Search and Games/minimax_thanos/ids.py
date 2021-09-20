import math
import time
from sys import stderr
from typing import Tuple, Optional, List

from fishing_game_core.game_tree import Node
from fishing_game_core.shared import ACTION_TO_STR
from minimax_thanos.utils import MinimaxAgent, MinimaxAgentHParams, point_distance_l1, get_node_repr, diff_x


class IDSAgent(MinimaxAgent):
    """
    Augmentations from ABAgent:
    ✅  Iterative Deepening Search (IDS)
    ✅  Add timeout at 60ms
    ✅  Check for invalid moves (to further prune the tree)
    ❌  Reuse of results for reordering
    ❌  Add checks for 1 fish in heuristic
    """
    CHECK_REPEATED_STATES = False
    TOTAL_FISH_SCORE = 0  # sum of the positive scores
    TOTAL_FISH_SCORE_HALF = 0  # half of sum of the positive scores
    TOTAL_FISH_SCORE_WN = 0  # sum with negative scores
    INITIAL_HOOK_POS = None

    HP: Optional[MinimaxAgentHParams] = None

    def __init__(self, initial_data: dict, hparams: MinimaxAgentHParams):
        """
        Initialize agent using the given the data containing the fish scores and types.
        :param dict initial_data:
        """
        # Set total scores available in game
        for k, v in initial_data.items():
            if k.startswith('fish'):
                fish_score = v['score']
                self.TOTAL_FISH_SCORE += fish_score if fish_score > 0 else 0
                self.TOTAL_FISH_SCORE_WN += fish_score
        self.TOTAL_FISH_SCORE_HALF = self.TOTAL_FISH_SCORE // 2
        # Extract and save agent settings
        self.HP = hparams
        # Visited States
        self.EXPLORED_SET = {}
        # Timer functionality
        self.time_start = 0.
        #print("IDS AGENT Created" * 100)

    def heuristic(self, node: Node) -> float:
        our_hook_pos, opp_hook_pos = node.state.hook_positions.values()
        our_score, opp_score = node.state.player_scores.values()
        fishes_count = len(node.state.fish_positions)

        # Add the caught fish to the scores
        if node.state.player_caught[0] != -1:
            our_score += node.state.fish_scores[node.state.player_caught[0]]
        if node.state.player_caught[1] != -1:
            opp_score += node.state.fish_scores[node.state.player_caught[1]]


        # Check if state is a terminating one
        if fishes_count == 0:
            return (our_score - opp_score) * self.HP.PLAYER_SCORE_MULTIPLIER * 10

        # Check if state is a losing one
        if opp_score > self.TOTAL_FISH_SCORE_HALF:
            return -math.inf
        # Check if state is a winning one
        if our_score > self.TOTAL_FISH_SCORE_HALF:
            return +math.inf

        return sum([
            #   - encourage winning, prefer preventing the enemy from getting points
            (our_score - 100 * opp_score) * 10,
            #   - discourage collisions
            # point_distance_l1(MinimaxModel.INITIAL_HOOK_POSITION, hook_pos) * 1,
            #   - encourage positions in the vicinity of closest fish
            -min([point_distance_l1(our_hook_pos, fp) - 2 * node.state.fish_scores[fi]
                  for fi, fp in node.state.fish_positions.items()] if len(node.state.fish_positions) else [0, ]) * 2,
            # 2 * random.random(),
        ])

    def minimax(self, node: Node, player: int, depth: int, alpha: Optional[float] = None,
                beta: Optional[float] = None, node_repr=None) -> Tuple[int, float]:
        # 0. Check timer
        if time.time() - self.time_start > self.HP.TIMEOUT_DURATION:
            raise TimeoutError
        # 1. Repeated states checking
        if IDSAgent.CHECK_REPEATED_STATES:
            if node_repr is None:
                node_repr = get_node_repr(node=node)
                if node_repr in self.EXPLORED_SET:
                    return self.EXPLORED_SET[node_repr]
        # 2. Get all children (plus Move Reordering)
        #children: List[Node] = sorted(node.compute_and_get_children(), key=self.heuristic, reverse=True)
        children: List[Node] = node.compute_and_get_children()
        # 3. Check if reached leaf nodes or max depth
        if len(children) == 0 or depth == 0:
            return node.move, self.heuristic(node=node)

        # # 3.1. Remove "stay" move if still for too many moves
        # if node.depth == 0 and MOVES_STILL >= MOVES_STILL_THRESHOLD:
        #     to_remove = None
        #     child: Node
        #     for i, child in enumerate(children):
        #         if child.move == 0:
        #             to_remove = i
        #             break
        #     if to_remove is not None:
        #         del children[to_remove]

        # 3.2. Remove "up" move if not valid
        our_hook_pos = node.state.hook_positions[0]
        opp_hook_pos = node.state.hook_positions[1]
        if our_hook_pos[1] == 19 and node.state.player_caught[0] == -1:
            to_remove = None
            child: Node
            for i, child in enumerate(children):
                if child.move == 1:
                    to_remove = i
                    break
            if to_remove is not None:
                del children[to_remove]

        # 3.3. Remove left/right move
        hook_dist_x, dist_min_idx, dist_right_idx = diff_x(our_hook_pos[0], opp_hook_pos[0])
        if 1 == hook_dist_x:
            to_remove = None
            child: Node
            for i, child in enumerate(children):
                if child.move == 4 and (dist_min_idx + dist_right_idx == 1) or \
                        child.move == 3 and (dist_min_idx * dist_right_idx == 1 or dist_min_idx + dist_right_idx == 0):
                    to_remove = i
                    break
            if to_remove is not None:
                del children[to_remove]

        # 4. Recurse
        # 4.1 MAX player
        if player == 0:
            argmax = 0
            max_value = -math.inf
            children_values = []
            children = sorted(children, key=self.heuristic, reverse=True)
            for i, child in enumerate(children):
                m, v = self.minimax(node=child, player=1, alpha=alpha, beta=beta, depth=depth - 1)
                children_values.append(v)
                if v > max_value:
                    max_value = v
                    argmax = i
                alpha = max(alpha, max_value)
                if beta <= alpha:
                    break
            #if depth == 8:
                #print(children_values)
            # Store node in the explored set (Graph Version)
            self.EXPLORED_SET[node_repr] = (children[argmax].move, max_value)
            # IDS_VALUES[node_repr] = max_value
            return children[argmax].move, max_value
        # 4.2. MIN player
        else:
            argmin = 0
            min_value = math.inf
            children = sorted(children, key=self.heuristic, reverse=False)
            for i, child in enumerate(children):
                m, v = self.minimax(node=child, player=0, alpha=alpha, beta=beta, depth=depth - 1)
                #   - find min value of children (rational opponent)
                if v < min_value:
                    min_value = v
                    argmin = i
                beta = min(beta, min_value)
                if beta <= alpha:
                    break
            # Store node in the explored set (Graph Version)
            self.EXPLORED_SET[node_repr] = (children[argmin].move, min_value)
            # IDS_VALUES[node_repr] = min_value
            return children[argmin].move, min_value

    # noinspection PyBroadException
    def get_next_move(self, initial_node: Node) -> Tuple[int, float]:
        # Record starting time
        self.time_start = time.time()
        # Record initial hook position
        # self.INITIAL_HOOK_POS = initial_node.state.hook_positions[0]
        # Add initial state in the explored set to avoid big loops
        root_node_repr = get_node_repr(initial_node)
        # Run Minimax at incrementing depth and return best move
        final_mm_move = 0
        # old_mm_value = -math.inf
        d = 0
        for d in range(1, self.HP.MAX_DEPTH, 1):
            print("Checking depth " , d , file=stderr)
            self.EXPLORED_SET[root_node_repr] = (0, -math.inf)
            try:
                final_mm_move, _ = self.minimax(node=initial_node, player=0, depth=d, alpha=-math.inf, beta=math.inf,
                                                node_repr=root_node_repr)
                # if mm_value > old_mm_value:
                #     final_mm_move = mm_move
                #     old_mm_value = mm_value
            except:
                print("Maximum depth reached " , d - 1, file=stderr)
                break
        if final_mm_move is None:
            print('\t> None --> 0', file=stderr)
            final_mm_move = 0

        # # Force it to move!
        # if final_mm_move == 0 and MOVES_STILL > MOVES_STILL_THRESHOLD:
        #     print('\t>>> Forcing it to move...', file=stderr)
        #     return ACTION_TO_STR[random.randint(2, 4)]
        #
        # if final_mm_move in (3, 4):
        #     MOVES_STILL = 0
        # else:
        #     MOVES_STILL += 1
        # OPPOSITE_OF_PREVIOUS_MOVE = OPPOSITE_OF_MOVE[final_mm_move]
        #print(f'---> MOVE = {ACTION_TO_STR[final_mm_move]}', file=stderr)
        return final_mm_move, 0.
