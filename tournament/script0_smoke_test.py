"""
Script used to check the entries of the participants. This is done by importing and instantiating each submitted
agent and verifying that everything loads correctly.

The participants of the tournament should be specified in the 'participants.cfg'
file. The corresponding PlayerAgent should be located in the 'submissions.group_N.PlayerAgent' module for each
participant entered in the tournament.
"""

import sys
import importlib
import traceback
from pathlib import Path
from TronGame import time_limit

# Ensure that the project root is found in PATH.
sys.path.insert(0, str(Path('..').resolve()))


def main():
    # Read participants from the file.
    with open('participants.cfg', 'r') as f:
        participants = [int(x.strip().split(',')[0]) for x in f.readlines()]
    print('Participants:', participants)

    bad_submissions = []
    for team_num in participants:
        print(f' -----> PROCESSING TEAM [{team_num}] <-----')
        try:
            with time_limit(5):
                agent_class = importlib.import_module(f'submissions.group_{team_num}.PlayerAgent').PlayerAgent
                agent = agent_class(0)
            print('Successfully imported and instantiated PlayerAgent.')
        except:
            bad_submissions.append(team_num)
            print(traceback.format_exc())
        print('========================================================================================')

    print('The following submissions have errors:', bad_submissions)


if __name__ == '__main__':
    main()
