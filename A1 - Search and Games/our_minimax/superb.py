import math
import time
# from sys import stderr
from typing import Tuple, Optional, List

from fishing_game_core.game_tree import Node
from our_minimax.utils import MinimaxAgent, MinimaxAgentHParams, point_distance_l1, get_node_repr, diff_x


class PVSAgent(MinimaxAgent):
    """
    Augmentations from IDSAgent:
    ✅  Implement Principal variation search (PVS) since we acknowledge that we use a strong heuristic
        (src: http://www.fierz.ch/strategy2.htm#depthreduction)
    ✅  Add depth to repeated states checking
    ✅  Add minimax value to node after calculation (used ONLY for children re-ordering)
    ✅  Add heuristic value to node after calculation (always used)
    ✅  Early-return from heuristic()
    ✅  Clean up code for final submission
    """
    TOTAL_FISH_SCORE = 0  # sum of the positive scores
    TOTAL_FISH_SCORE_HALF = 0  # half of sum of the positive scores
    TOTAL_FISH_SCORE_WN = 0  # sum with negative scores

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

    def heuristic(self, node: Node) -> float:
        """
        Heuristic function to approximate utility function for a given state (wrapped in game_tree.Node instance).
        :param Node node: a game_tree.Node instance
        :return: a float object containing calculated heuristic value for the node
        """
        # Check if value has already been calculated before
        state = node.state
        if hasattr(state, 'h_value'):
            return state.h_value

        # Add the caught fish to the scores
        our_score, opp_score = state.player_scores.values()
        if state.player_caught[0] != -1:
            our_score += state.fish_scores[state.player_caught[0]]
        if state.player_caught[1] != -1:
            opp_score += state.fish_scores[state.player_caught[1]]

        # Unwrap hook position variables
        our_hook_pos, opp_hook_pos = state.hook_positions.values()

        # Check if state is a terminating one
        fishes_count = len(state.fish_positions)
        if fishes_count == 0:
            return_value = (our_score - opp_score) * self.HP.PLAYER_SCORE_MULTIPLIER * 10

        # Check if state is a losing one
        elif opp_score > self.TOTAL_FISH_SCORE_HALF:
            return_value = -math.inf
        # Check if state is a winning one
        elif our_score > self.TOTAL_FISH_SCORE_HALF:
            return_value = +math.inf
        # Else: return a weighted avg of the difference of scores at the given node plus the minimum distance from all
        # fishes (we subtract fish score so as for negative-scored fishes to appear more away
        else:
            # Sum the following encouragements
            #   - encourage winning, prefer preventing the enemy from getting points
            #   - encourage positions in the vicinity of closest fish
            return_value = (our_score - opp_score) * self.HP.PLAYER_SCORE_MULTIPLIER - \
                           min([point_distance_l1(our_hook_pos, fp) - 2 * state.fish_scores[fi]
                                for fi, fp in state.fish_positions.items()]) * 2

        # Save computed value in node
        node.h_value = return_value
        return return_value

    def minimax(self, node: Node, player: int, depth: int, alpha: Optional[float] = None,
                beta: Optional[float] = None, node_repr=None) -> Tuple[int, float]:
        # 0. Check timer
        if time.time() - self.time_start > self.HP.TIMEOUT_DURATION:
            raise TimeoutError
        # 1. Repeated states checking
        if node_repr is None:
            node_repr = get_node_repr(node=node)
            if node_repr in self.EXPLORED_SET and node.depth >= self.EXPLORED_SET[node_repr][2]:
                return self.EXPLORED_SET[node_repr][:2]
        # 2. Get all children
        children: List[Node] = node.compute_and_get_children()
        # 3. Check if reached leaf nodes or max depth
        if len(children) == 0 or depth == 0:
            h_value = self.heuristic(node=node)
            self.EXPLORED_SET[node_repr] = (node.move, h_value, node.depth)
            return node.move, h_value

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

        our_hook_pos = node.state.hook_positions[0]
        opp_hook_pos = node.state.hook_positions[1]
        player_index = node.state.player
        invalid_moves = set()

        # 3.2. Remove "up" move if not valid
        if node.state.hook_positions[player_index][1] == 19:
            invalid_moves.add(1)

        # 3.3. Remove left/right move if not valid
        # Based on the outputs of diff_x:
        #      dist, min_idx, right_idx = diff_x(our_hook, opp_hook)
        # where if dist==1, then we forbid the move based on the following table.
        #  ___________________________________________________
        # |  min_idx  |  right_idx  |  forbidden_move_for_us  |
        #  ---------------------------------------------------
        # |     0     |      0      |          left           |
        # |     0     |      1      |          right          |
        # |     1     |      0      |          right          |
        # |     1     |      1      |          left           |
        #  ---------------------------------------------------
        # In case of opponent's turn, the forbidden moves are vice-versa.
        hook_dist_x, dist_min_idx, dist_right_idx = diff_x(our_hook_pos[0], opp_hook_pos[0])
        if 1 == hook_dist_x:
            if dist_min_idx + dist_right_idx == 1:
                invalid_moves.add(4 - player_index)
            if dist_min_idx * dist_right_idx == 1 or dist_min_idx + dist_right_idx == 0:
                invalid_moves.add(3 + player_index)

        # 3.4. Remove "down" move if not valid
        if node.state.hook_positions[player_index][1] <= 1:
            invalid_moves.add(2)

        # 3.5. Filter children
        children = [c for c in children if c.move not in invalid_moves]
        #  - recheck children length
        if len(children) == 0:
            h_value = self.heuristic(node)
            self.EXPLORED_SET[node_repr] = (node.move, h_value, node.depth)
            return 0, h_value

        # 4. Recurse
        # 4.1 MAX player
        if player == 0:
            argmax = 0
            max_value = -math.inf
            children = sorted(children, key=lambda n: n.h_value if hasattr(n, 'h_value') else self.heuristic(n),
                              reverse=True)
            for i, child in enumerate(children):
                if i == 0:
                    # Consider first child as the Principal Variation component (i.e. assuming we achieved perfect
                    # re-ordering)
                    m, v = self.minimax(node=child, player=1, alpha=alpha, beta=beta, depth=depth - 1)
                else:
                    m, v = self.minimax(node=child, player=1, alpha=alpha, beta=alpha + 1, depth=depth - 1)
                    if alpha < v < beta:
                        # PVS failed, do a full re-search
                        m, v = self.minimax(node=child, player=1, alpha=v, beta=beta, depth=depth - 1)
                if v > max_value:
                    max_value = v
                    argmax = i
                alpha = max(alpha, max_value)
                if beta <= alpha:
                    break
            # Store node in the explored set (Graph Version)
            self.EXPLORED_SET[node_repr] = (children[argmax].move, max_value, node.depth)
            # node.mm_value = max_value
            return children[argmax].move, max_value
        # 4.2. MIN player
        else:
            argmin = 0
            min_value = math.inf
            children = sorted(children, key=lambda n: n.h_value if hasattr(n, 'h_value') else self.heuristic(n),
                              reverse=False)
            # Prune opponent's tree
            if len(children) > 2:
                children = children[:2]
            for i, child in enumerate(children):
                if i == 0:
                    # Consider first child as the Principal Variation component (i.e. assuming we achieved perfect
                    # re-ordering)
                    m, v = self.minimax(node=child, player=0, alpha=alpha, beta=beta, depth=depth - 1)
                else:
                    m, v = self.minimax(node=child, player=0, alpha=alpha, beta=alpha + 1, depth=depth - 1)
                    if alpha < v < beta:
                        # PVS failed, do a full re-search
                        m, v = self.minimax(node=child, player=0, alpha=v, beta=beta, depth=depth - 1)
                #   - find min value of children (rational opponent)
                if v < min_value:
                    min_value = v
                    argmin = i
                beta = min(beta, min_value)
                if beta <= alpha:
                    break
            # Store node in the explored set (Graph Version)
            self.EXPLORED_SET[node_repr] = (children[argmin].move, min_value, node.depth)
            # node.mm_value = min_value
            return children[argmin].move, min_value

    # noinspection PyBroadException
    def get_next_move(self, initial_node: Node) -> int:
        # Early return
        if initial_node.state.player_caught[0] != -1:
            return 1
        # Record starting time
        self.time_start = time.time()
        # Add initial state in the explored set to avoid big loops
        root_node_repr = get_node_repr(initial_node)
        # Run Minimax at incrementing depth and return best move
        mm_move = None
        for d in range(1, self.HP.MAX_DEPTH):
            self.EXPLORED_SET = {root_node_repr: (0, -math.inf, 0)}
            try:
                mm_move, mm_value = self.minimax(node=initial_node, player=0, depth=d, alpha=-math.inf, beta=math.inf,
                                                 node_repr=root_node_repr)
                # If return minimax value was inf, just return (winning move)
                if mm_value == math.inf:
                    return mm_move
            except TimeoutError:
                # print("Maximum depth reached ", d - 1, file=stderr)
                break
        # print('', file=stderr)
        # print(f'<<<<<<<<<<<<<<<<<<<< MOVE ("{ACTION_TO_STR[mm_move]}") DONE >>>>>>>>>>>>>>>>>>>>', file=stderr)
        # print('', file=stderr)
        return mm_move or 0
