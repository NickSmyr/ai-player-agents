from typing import Tuple, Optional, List

from fishing_game_core.game_tree import Node
from minimax_thanos.utils import MinimaxAgent, MinimaxAgentHParams, point_distance_l1


class NaiveAgent(MinimaxAgent):
    """
    Initial implementation of Minimax S/W Agent. As heuristic function we just use the difference of player scores and
    the minimum fish distance.
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

    def heuristic(self, node: Node) -> float:
        our_hook_pos, opp_hook_pos = node.state.hook_positions.values()
        our_score, opp_score = node.state.player_scores.values()
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
                beta: Optional[float] = None) -> Tuple[int, float]:
        # 1. Get all children
        children: List[Node] = node.compute_and_get_children()
        # 2. Check if reached leaf nodes or max depth
        if len(children) == 0 or depth == 0:
            return node.move, self.heuristic(node=node)
        # 3. Recurse
        children_values = [self.minimax(node=child, player=1 - player, depth=depth - 1)[1] for child in children]
        children_values_len = len(children_values)
        if depth == 3:
            print(children_values)
        # 3.1. MAX player
        if player == 0:
            #   - find max value of children
            argmax = max(range(children_values_len), key=lambda v: children_values[v])
            max_value = children_values[argmax]
            #   - in case of equal value, select the move <> 0
            if node.depth == 0:
                for i in range(argmax, children_values_len):
                    if children_values[i] == max_value and children[i].move != 0:
                        # if depth == 3:
                        #     print(children[i].move, max_value)
                        return children[i].move, max_value
            # if depth == 3:
            #     print(children[argmax].move, max_value)
            return children[argmax].move, max_value
        # 3.2. MIN player
        argmin = min(range(len(children_values)), key=lambda v: children_values[v])
        return children[argmin].move, children_values[argmin]

    def get_next_move(self, initial_node: Node) -> Tuple[int, float]:
        self.INITIAL_HOOK_POS = initial_node.state.hook_positions[0]
        return self.minimax(node=initial_node, player=0, depth=self.HP.MAX_DEPTH)
