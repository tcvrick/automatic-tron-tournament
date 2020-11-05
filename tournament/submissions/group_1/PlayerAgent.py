import numpy as np


class PlayerAgent:

    def __init__(self, player_id):
        self.player_id = player_id

    def choose_move(self, game):
        # Get an array of legal moves from your current position.
        legal_moves = game.get_legal_moves(self.player_id)

        # Shuffle the legal moves and pick the first one. This is equivalent
        # to choosing a move randomly with no logic.
        np.random.shuffle(legal_moves)
        return legal_moves[0]
