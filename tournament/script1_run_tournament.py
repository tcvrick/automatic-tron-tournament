"""
Script used to run the tournament. The participants of the tournament should be specified in the 'participants.cfg'
file. The corresponding PlayerAgent should be located in the 'submissions.group_N.PlayerAgent' module for each
participant entered in the tournament.

You can run the 'smoke_test' script to verify everything is in place beforehand.
"""

import sys
import time
import pickle
import traceback
import importlib
import platform
import numpy as np
import pandas as pd

from tabulate import tabulate
from collections import OrderedDict
from tinydb import TinyDB, Query
from trueskill import Rating, quality_1vs1, rate_1vs1
from multiprocessing import Process, Queue, cpu_count
from multiprocessing.queues import Empty
from collections import namedtuple
from pathlib import Path
from TronGame import TronGame

# Global Settings
if platform.system() == 'Windows':
    MAX_NUM_PROCESSES = 12  # PC
else:
    MAX_NUM_PROCESSES = 24  # Google Cloud
DATABASE_FILENAME = 'tournament_db.json'

# Ensure that the project root is found in PATH.
sys.path.insert(0, str(Path('..').resolve()))


def main():
    # ==================================================================================================================
    # 1. SET-UP
    # ==================================================================================================================
    print('===========================================================================================================')
    print('INITIALIZATION')
    print('===========================================================================================================')
    # Read participants from the config file.
    with open('participants.cfg', 'r') as f:
        participant_team_ids = [int(x.strip().split(',')[0]) for x in f.readlines()]
    with open('participants.cfg', 'r') as f:
        participant_team_names = [x.strip().split(',')[1] for x in f.readlines()]

    participant_team_names = dict(zip(participant_team_ids, participant_team_names))
    print('Found these participants in the config file:', participant_team_ids, participant_team_names)

    # Import the agents of the participants.
    participant_agent_classes = dict(zip(participant_team_ids,
                                         [load_participant(x) for x in participant_team_ids]))

    # Define a data-structure which contains the relevant information for each game played.
    match_data_fields = ['match_id', 'player1_id', 'player2_id', 'winner', 'board_history']
    MatchData = namedtuple('MatchData', field_names=match_data_fields)

    # Load the database.
    db = TinyDB(DATABASE_FILENAME)
    history_table = db.table('match_history')
    history_query = Query()
    mmr_table = db.table('mmr')
    mmr_query = Query()

    # Initialize the match IDs.
    starting_match_id = max([0] + [item['match_id'] for item in history_table.search(history_query['match_id'])])
    print(f'Initializing matches starting from [{starting_match_id}]')

    # Initialize the MMRs.
    mmr = OrderedDict()
    for team_id in participant_team_ids:
        q = mmr_table.search(mmr_query['player_id'] == team_id)
        if q:
            q = q[0]
            print(f'Initialized player [{team_id}] from the existing database... mu={q["mu"]}, sigma={q["sigma"]}')
            mmr[team_id] = Rating(mu=q["mu"], sigma=q["sigma"])
        else:
            mmr_table.insert({'player_id': team_id, 'mu': 100, 'sigma': 8.333})
            mmr[team_id] = Rating(mu=100, sigma=8.333)

    # ==================================================================================================================
    # 2. RUN TOURNAMENT
    # ==================================================================================================================
    print('===========================================================================================================')
    print('TOURNAMENT START')
    print('===========================================================================================================')
    active_matches = {}
    num_matches = starting_match_id
    queue = Queue()
    while True:

        # --------------------------------------------------------------------------------------------------------------
        # 2.1 MATCHMAKING
        # --------------------------------------------------------------------------------------------------------------
        while len(active_matches) < MAX_NUM_PROCESSES:
            # Record keeping.
            num_matches += 1
            match_id = num_matches

            # Matchmaking system. Randomly select two players for the next match; players with similar MMRs are
            # more likely to be paired together.
            player1_id, player2_id = get_matchup(participant_team_ids, mmr)

            # Start the game in a separate Python process. This subprocess will place the results of the game into
            # a shared queue once its finished (which will be processed later).
            p = Process(target=play_tournament_game, args=[match_id,
                                                           participant_agent_classes[player1_id],
                                                           participant_agent_classes[player2_id],
                                                           queue])

            print(f'[MATCH - {match_id}] Starting match between Player [{player1_id}] and Player [{player2_id}]!')
            p.daemon = True
            p.start()
            assert match_id not in active_matches.keys()
            active_matches[match_id] = (match_id, player1_id, player2_id, p, time.time())

        # --------------------------------------------------------------------------------------------------------------
        # 2.2 MAINTENANCE
        # --------------------------------------------------------------------------------------------------------------
        while True:
            # Check if the processes has put any results into the queue yet, if there are no results, sleep the
            # main thread and check back later.
            try:
                match_id, winner, board_history = queue.get_nowait()
            except Empty:
                break

            # If there is an item in the queue, remove the match from the active matches dictionary and join
            # the sub-process. Finally, update the MMR/match history as appropriate and save to file.
            _, p1_id, p2_id, process, time_start = active_matches[match_id]
            process.join()
            del active_matches[match_id]

            # Save the match to file.
            print(f'[MATCH - {match_id}] Finished match between Player [{p1_id}] and '
                  f'Player [{p2_id}], Player [{[p1_id, p2_id][winner - 1]}] wins! '
                  f'Elapsed time: [{time.time() - time_start}s]')

            # Update the MMRs based on the winner. Note that the winner ID may be -1 if there was an exception
            # during the playing of the game. In this case, ignore the game and pretend it didn't happen.
            if winner == 1:
                winner_id, loser_id = p1_id, p2_id
            elif winner == 2:
                winner_id, loser_id = p2_id, p1_id
            else:
                continue

            winner_rating, loser_rating = rate_1vs1(mmr[winner_id], mmr[loser_id])
            mmr[winner_id], mmr[loser_id] = winner_rating, loser_rating

            # Only periodically compute the player statistics/standings to save on compute time.
            if match_id % 10 == 0:
                print('-----------------------------------------------------------------------------------------------')
                print('MMR STANDINGS')
                print('-----------------------------------------------------------------------------------------------')
                sorted_mmrs = OrderedDict(sorted(mmr.items(), key=lambda x: x[1].mu - (2 * x[1].sigma), reverse=True))
                table_data = [
                    (i + 1,
                     f'{k} - {participant_team_names[k]}',
                     # k,
                     round(v.mu - (2 * v.sigma), ndigits=2),
                     len(history_table.search(
                         ((history_query['player1_id'] == k) & (history_query['winner'] == 1)) |
                         ((history_query['player2_id'] == k) & (history_query['winner'] == 2)))),
                     len(history_table.search(
                         (history_query['player1_id'] == k) | (history_query['player2_id'] == k))),
                     )
                    for i, (k, v) in enumerate(sorted_mmrs.items())]
                print(tabulate(table_data, headers=['Ranking', 'Team ID', 'MMR', 'Games Won', 'Games Played'],
                               tablefmt='fancy_grid', stralign='right'))
                print('-----------------------------------------------------------------------------------------------')

            # Save MMR updates to database.
            mmr_table.update({'mu': winner_rating.mu, 'sigma': winner_rating.sigma},
                             mmr_query['player_id'] == winner_id)
            mmr_table.update({'mu': loser_rating.mu, 'sigma': loser_rating.sigma},
                             mmr_query['player_id'] == loser_id)

            # Save match history updates to database.
            history_table.insert({'match_id': int(match_id), 'player1_id': int(p1_id), 'player2_id': int(p2_id),
                                  'winner': int(winner)})

        time.sleep(0.1)


def play_tournament_game(match_id, agent1_class, agent2_class, pipe):
    tron_game = None
    try:
        tron_game = TronGame(agent1_class=agent1_class,
                             agent2_class=agent2_class,
                             board_size=20,
                             board_type='rocky',
                             match_id=match_id)

        (player1_wins, player2_wins), board_history = tron_game.play_series(best_of_n_games=1)
        winner = np.argmax([player1_wins, player2_wins]) + 1
        pipe.put((match_id, winner, board_history))
    except:
        if tron_game:
            # TODO: Log to separate file.
            print(f'[MATCH ID: {tron_game.match_id}]', traceback.format_exc())
        pipe.put((match_id, -1, np.array([])))


def load_participant(team_id: int):
    agent = importlib.import_module(f'submissions.group_{team_id}.PlayerAgent').PlayerAgent
    return agent


def get_matchup(participant_team_ids, mmr):
    # Choose the first player randomly.
    player1_id = np.random.choice(participant_team_ids, size=1, replace=False)

    # Compute the probability the chosen player should face the other remaining teams. Compute the probabilities
    # such that players closer to each other in MMR are more likely to play each other.
    other_players = list(participant_team_ids)
    other_players.remove(player1_id)

    # The faceoff probability is computed based on the 'quality_1vs1' method of the trueskill module. Perfectly
    # matched MMRs return a match quality of ~0.44, and sharply falls off to 0 as the MMRs diverge. To avoid having
    # the same teams play against each other continuously, a bias of ~0.1 is added to the weighting of each opponent.
    faceoff_probabilities = np.array([0.10 + quality_1vs1(mmr[int(player1_id)], mmr[int(x)])
                                      for x in other_players])
    faceoff_probabilities /= faceoff_probabilities.sum()
    player2_id = np.random.choice(other_players, size=1, replace=False, p=faceoff_probabilities)
    player1_id, player2_id = sorted([int(player1_id), int(player2_id)])

    return player1_id, player2_id


if __name__ == '__main__':
    main()
