#!/usr/bin/env python3

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR
# from minimax.naive import NaiveAgent
# from minimax.naive import ABAgent
# from our_minimax.ids import IDSAgent
from our_minimax.superb import PVSAgent
from our_minimax.utils import MinimaxAgentHParams


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
        # return NaiveAgent(initial_data, hparams=HParams({
        #     'MAX_DEPTH': 3,
        # }))
        # return ABAgent(initial_data, hparams=HParams({
        #     'MAX_DEPTH': 6,
        # }))
        # return IDSAgent(initial_data, hparams=MinimaxAgentHParams({
        #     'MAX_DEPTH': MAX_DEPTH_IDS,
        #     'TIMEOUT_DURATION': TIME_THRESHOLD,
        # }))
        return PVSAgent(initial_data, hparams=MinimaxAgentHParams({
            'MAX_DEPTH': 9,
            'TIMEOUT_DURATION': 60 * 1e-3,
        }))

    def search_best_next_move(self, model, initial_tree_node):
        return ACTION_TO_STR[model.get_next_move(initial_tree_node)]
