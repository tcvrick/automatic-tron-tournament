
class SimpleSearchAgent:
    SEARCH_DEPTH = 9

    def __init__(self, player_id):
        self.player_id = player_id

    def choose_move(self, game):
        moves = []
        self.path_find(game, game.board, [], moves, exit_flag=[])

        max_len = max([len(x) for x in moves])
        longest_moves = [x for x in moves if len(x) == max_len]
        return longest_moves[0][0]

    def path_find(self, game, current_board, current_path, paths_list, exit_flag):
        # Check for exit condition. If we run into a leaf node, terminate the current path and continue searching.
        # If we reach the maximum search depth, terminate the current path and set the EXIT_FLAG and terminate the
        # search.
        legal_moves = game.get_legal_moves(self.player_id, current_board)
        if exit_flag:
            paths_list.append(current_path)
            return
        elif not legal_moves:
            paths_list.append(current_path)
            return
        elif len(current_path) >= self.SEARCH_DEPTH:
            paths_list.append(current_path)
            exit_flag += [True]
            return

        # Evaluate the potential boards from this position and continue searching.
        for child_move in legal_moves:
            next_board = game.examine_move(self.player_id, child_move, current_board)
            self.path_find(game, next_board, current_path[:] + [child_move], paths_list, exit_flag)
