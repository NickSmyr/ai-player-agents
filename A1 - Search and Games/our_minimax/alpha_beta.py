import math
from typing import Tuple, Optional, List

from fishing_game_core.game_tree import Node
from our_minimax.utils import MinimaxAgent, MinimaxAgentHParams, point_distance_l1, get_node_repr


class ABAgent(MinimaxAgent):
    """
    Augmentations from NaiveAgent:
    ✅  Alpha-Beta Pruning
    ✅  Move Reordering
    ✅  Repeated States Checking (incl. initial node)
    ✅  Add killer moves in heuristic
    """
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

    def heuristic(self, node: Node) -> float:
        our_hook_pos, opp_hook_pos = node.state.hook_positions.values()
        our_score, opp_score = node.state.player_scores.values()
        fishes_count = len(node.state.fish_positions)

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
            #   - encourage winning
            (our_score - opp_score) * 10,
            #   - discourage collisions
            # point_distance_l1(MinimaxModel.INITIAL_HOOK_POSITION, hook_pos) * 1,
            #   - encourage positions in the vicinity of closest fish
            -min([point_distance_l1(our_hook_pos, fp) - 2 * node.state.fish_scores[fi]
                  for fi, fp in node.state.fish_positions.items()] if len(node.state.fish_positions) else [0, ]) * 2,
            # 2 * random.random(),
        ])

    def minimax(self, node: Node, player: int, depth: int, alpha: Optional[float] = None,
                beta: Optional[float] = None, node_repr=None) -> Tuple[int, float]:
        # 1. Repeated states checking
        if node_repr is None:
            node_repr = get_node_repr(node=node)
            if node_repr in self.EXPLORED_SET:
                return self.EXPLORED_SET[node_repr]
        # 2. Get all children (plus Move Reordering)
        children: List[Node] = sorted(node.compute_and_get_children(), key=self.heuristic, reverse=True)
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

        # # 3.2. Remove "up" move if not valid
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
        # # 3.3. Remove left/right move
        # hook_dist_x, dist_min_idx, dist_right_idx = diff_x(our_hook_pos[0], opp_hook_pos[0])
        # if 1 == hook_dist_x:
        #     to_remove = None
        #     child: Node
        #     for i, child in enumerate(children):
        #         if child.move == 4 and (dist_min_idx + dist_right_idx == 1) or \
        #                 child.move == 3 and (dist_min_idx * dist_right_idx == 1 or \
        #                                      dist_min_idx + dist_right_idx == 0):
        #             to_remove = i
        #             break
        #     if to_remove is not None:
        #         del children[to_remove]

        # 4. Recurse
        # 4.1 MAX player
        if player == 0:
            argmax = 0
            max_value = -math.inf
            children_values = []
            for i, child in enumerate(children):
                m, v = self.minimax(node=child, player=1, alpha=alpha, beta=beta, depth=depth - 1)
                children_values.append(v)
                if v > max_value:
                    max_value = v
                    argmax = i
                alpha = max(alpha, max_value)
                if beta <= alpha:
                    break
            if depth == 6:
                print(children_values)
            # Store node in the explored set (Graph Version)
            self.EXPLORED_SET[node_repr] = (children[argmax].move, max_value)
            # IDS_VALUES[node_repr] = max_value
            return children[argmax].move, max_value
        # 4.2. MIN player
        else:
            argmin = 0
            min_value = math.inf
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

    def get_next_move(self, initial_node: Node) -> Tuple[int, float]:
        # Record initial hook position
        self.INITIAL_HOOK_POS = initial_node.state.hook_positions[0]
        # Add initial state in the explored set to avoid big loops
        root_node_repr = get_node_repr(initial_node)
        self.EXPLORED_SET[root_node_repr] = (0, -math.inf)
        # Run Minimax and return best move
        return self.minimax(node=initial_node, player=0, depth=self.HP.MAX_DEPTH, alpha=-math.inf, beta=math.inf,
                            node_repr=root_node_repr)
