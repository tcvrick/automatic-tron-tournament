import matplotlib
import signal
import platform
import traceback
import sys
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from contextlib import contextmanager

# Ensure that the project root is found in PATH.
sys.path.insert(0, str(Path('..').resolve()))


# ===============================================================================
# Game Implementation
# ===============================================================================

class TronGame:
    TIME_LIMIT = 5

    def __init__(self, agent1_class, agent2_class, board_size, board_type, match_id=-1):
        # Default board.
        if board_type == 'default':
            self.size = board_size
        # Board with obstacles and a fixed size of 10x10.
        elif board_type == 'obstacles':
            self.size = 10
        elif board_type == 'rocky':
            self.size = board_size
        else:
            raise ValueError('Invalid board type.')

        # Build the game board.
        self.match_id = match_id
        self.board_type = board_type
        self.board = self.build_board(board_type)

        # Initialize the game state variables and set the values using the
        # 'reset_game()' method.
        self.reset_game()

        # Initialize our agents.
        self.agent1 = agent1_class(1)
        self.agent2 = agent2_class(3)

    def build_board(self, board_type):
        """
        This method takes a board_type: ['default', 'obstacles'] and returns a
        new board (NumPy matrix).
        """

        # Default board.
        if board_type == 'default':
            board = np.zeros((self.size, self.size))
            board[0, 0] = 1
            board[self.size - 1, self.size - 1] = 3
        # Board with obstacles and a fixed size of 10x10.
        elif board_type == 'obstacles':
            board = np.zeros((10, 10))
            board[1, 4] = 1
            board[8, 4] = 3
            board[3:7, 0:4] = 4
            board[3:7, 6:] = 4
        # Board with obstacles and a fixed size of 10x10.
        elif board_type == 'rocky':
            board = np.zeros((self.size, self.size))
            a = np.random.randint(2, size=(self.size, self.size))
            b = np.random.randint(2, size=(self.size, self.size))
            c = np.random.randint(2, size=(self.size, self.size))
            d = np.random.randint(2, size=(self.size, self.size))

            board = board + (a * b * c * d) * 4
            board[0, 0] = 1
            board[self.size - 1, self.size - 1] = 3
        else:
            raise ValueError('Invalid board type.')

        return board

    def reset_game(self):
        """
        Helper method which re-initializes the game state.
        """

        self.board = self.build_board(self.board_type)

    def get_player_position(self, player_id, board=None):
        """
        Helper method which finds the coordinate of the specified player ID
        on the board.
        """

        if board is None:
            board = self.board
        coords = np.asarray(board == player_id).nonzero()
        coords = np.stack((coords[0], coords[1]), 1)
        coords = np.reshape(coords, (-1, 2))
        return coords[0]

    def get_legal_moves(self, player, board=None):
        """
        This method returns a list of legal moves for a given player ID and
        board.
        """

        if board is None:
            board = self.board

        # Get the current player position and then check for all possible
        # legal moves.
        y, x = self.get_player_position(player, board)
        moves = []

        # Up
        if (y != 0) and (board[y - 1, x] == 0):
            moves.append([y - 1, x])
        # Down
        if (y != self.size - 1) and (board[y + 1, x] == 0):
            moves.append([y + 1, x])
        # Left
        if (x != 0) and (board[y, x - 1] == 0):
            moves.append([y, x - 1])
        # Right
        if (x != self.size - 1) and (board[y, x + 1] == 0):
            moves.append([y, x + 1])

        return moves

    def examine_move(self, player, coordinate, board):
        board_clone = board.copy()
        prev = self.get_player_position(player, board_clone)
        board_clone[prev[0], prev[1]] = 4
        board_clone[coordinate[0], coordinate[1]] = player
        return board_clone

    @staticmethod
    def view_game(board_history):
        """
        This is a helper function which takes a board history
        (i.e., a list of board states) and creates an animation of the game
        as it progresses.
        """

        fig, ax = plt.subplots()
        colors = ['black', 'blue', 'pink', 'white', 'red', 'yellow']
        cmap = matplotlib.colors.ListedColormap(colors)
        bounds = np.linspace(0, 5, 6)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        matrix = ax.matshow(board_history[0], cmap=cmap, norm=norm)

        def update(i):
            matrix.set_array(board_history[i])
            return matrix

        ani = FuncAnimation(fig, update, frames=len(board_history), interval=100)
        plt.show()
        return HTML(ani.to_html5_video())

    def play_series(self, best_of_n_games: int, verbose=False):
        """
        This method plays a series of games between the two agents.

        It returns two objects: (i) a tuple which indicates the number of
        wins per player, and (ii) a history of the board state as the game
        progresses.
        """

        if not best_of_n_games & 1:
            raise ValueError('Number of games must be odd.')
        num_games_to_win = (best_of_n_games // 2) + 1

        wins_player_1 = 0
        wins_player_2 = 0
        games = []
        for i in range(best_of_n_games):
            winning_player_id, board_history = self.__play_game()
            games.append(board_history)

            if winning_player_id == 1:
                wins_player_1 += 1
            elif winning_player_id == 2:
                wins_player_2 += 1
            else:
                raise ValueError('Invalid winning player ID.')

            if wins_player_1 >= num_games_to_win or wins_player_2 >= num_games_to_win:
                break

        if verbose:
            print(f'Finished playing [{len(games)}] games, in a best of [{best_of_n_games}] series.')
            print(f'Player 1 won [{wins_player_1}] games and has a win-rate of [{wins_player_1 / len(games) * 100}%].')
            print(f'Player 2 won [{wins_player_2}] games and has a win-rate of [{wins_player_2 / len(games) * 100}%].')
        return (wins_player_1, wins_player_2), games

    def __apply_move(self, player, coordinate):
        """
        This private method moves a player ID to a new coordinate and obstructs
        the previously occupied tile.
        """

        prev_coord = self.get_player_position(player)

        self.board[prev_coord[0], prev_coord[1]] = 4
        self.board[coordinate[0], coordinate[1]] = player

    def __play_game(self):
        """
        This private method plays a single game between the two agents. It
        returns the winning player ID as well as the history of the board
        as the game progresses.
        """

        # Reset the game.
        self.reset_game()
        board_history = []

        # Play the game until it's conclusion.
        while True:
            # ---------------------------------------
            # PLAYER 1's TURN
            # ---------------------------------------
            # Check legal moves.
            legal_moves = self.get_legal_moves(1)
            if not len(legal_moves):
                winning_player_id = 2
                break

            # Compute and apply the chosen move.
            with time_limit(TronGame.TIME_LIMIT):
                try:
                    board_clone = self.board.copy()
                    move = self.agent1.choose_move(self)
                    if move not in legal_moves:
                        raise ValueError(f'Illegal move [{move}] returned from Player [1]')
                    if not np.all(board_clone == self.board):
                        raise ValueError(f'Illegal operation from Player [1], the state of the game board was changed.')
                except Exception:
                    # TODO: Log to separate file.
                    print(f'[MATCH ID: {self.match_id}]', traceback.format_exc())
                    winning_player_id = 2
                    break
            self.__apply_move(1, move)

            # Record keeping.
            board_history.append(np.array(self.board.copy()))

            # ---------------------------------------
            # PLAYER 2's TURN
            # ---------------------------------------
            # Check legal moves.
            legal_moves = self.get_legal_moves(3)
            if not len(legal_moves):
                winning_player_id = 1
                break

            # Compute and apply the chosen move.
            with time_limit(TronGame.TIME_LIMIT):
                try:
                    board_clone = self.board.copy()
                    move = self.agent2.choose_move(self)
                    if move not in legal_moves:
                        raise ValueError(f'Illegal move [{move}] returned from Player 2')
                    if not np.all(board_clone == self.board):
                        raise ValueError(f'Illegal operation from Player [2], the state of the game board was changed.')
                except Exception:
                    # TODO: Log to separate file.
                    print(f'[MATCH ID: {self.match_id}]', traceback.format_exc())
                    winning_player_id = 1
                    break
            self.__apply_move(3, move)

            # Record keeping.
            board_history.append(np.array(self.board.copy()))

        return winning_player_id, board_history


# ===============================================================================
# Utility
# ===============================================================================

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")

    if platform.system() != 'Windows':
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
    try:
        yield
    finally:
        if platform.system() != 'Windows':
            signal.alarm(0)


class RandomAgent:

    def __init__(self, player_id):
        self.player_id = player_id

    def choose_move(self, game):
        # Get an array of legal moves from your current position.
        legal_moves = game.get_legal_moves(self.player_id)

        # Shuffle the legal moves and pick the first one. This is equivalent
        # to choosing a move randomly with no logic.
        np.random.shuffle(legal_moves)
        return legal_moves[0]


# ===============================================================================
# Debug
# ===============================================================================
def main():
    from agents.SimpleSearchAgent import SimpleSearchAgent
    from submissions.group_1.PlayerAgent import PlayerAgent as Agent1
    SimpleSearchAgent.SEARCH_DEPTH = 20
    my_tron_game = TronGame(board_size=20,
                            agent1_class=Agent1,
                            agent2_class=SimpleSearchAgent,
                            board_type='rocky')

    (player1_wins, player2_wins), game_histories = my_tron_game.play_series(best_of_n_games=3, verbose=True)
    TronGame.view_game(game_histories[0])


if __name__ == '__main__':
    main()
