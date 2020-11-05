# Automated TRON Tournament

This project contains a framework for running an automatic TRON tournament (alongside an implementation of the game itself). Participants can enter an agent 
into the tournament by providing code which comforms to a re

## Setup

- **Config File** - Each line in the ```[tournament/participants.cfg]``` file specifies the team ID of
a participant in the tournament.

- **Match History** -  Stored in the ```[tournament/match_history.csv]``` file.

## Quickstart

- Configure the ```tournament/participants.cfg``` as desired. Each line represents the team number of a
participant. The corresponding PlayerAgent module should be placed in ```submissions/group_N/PlayerAgent.py```.

- Run the ```tournament/script0_smoke_test.py``` script to ensure that all participants are being loaded
properly.

- Run the ```tournament/script1_run_tournament.py``` script to play the tournament. The matches being started/finished,
as well as the MMRs should be constantly being outputted. Errors will also be dumped to ```stdout``` with
a corresponding match ID.