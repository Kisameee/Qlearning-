# Reinforcement learning project for 5IBD ESGI 2018-2019
This project aim to teach us reinforcement learning by implementing the popular RL models.


## TODO
- Better docstrings (understand '???' comments)
- Better type hints (find type of 'Any' type hints)
- Transform get methods in python property
- Tests everything
- Profile and rework heavy functions
  - WindJammersGameState.compute_current_score_and_end_game_more_efficient
  - WindJammersGameState.frisbee_hitplayer1
  - WindJammersGameState.frisbee_hitplayer2
- Comments Windjamers game
- Clean game_runner.py and reinforcement_battle.py


## How to use ?
You have to use the reinforcement_battle.py in command line.

Example :

    python -O reinforcement_battle.py Random Random TicTacToe 100 --no-gpu

Will run 100 games of TicTacToe with a Random agent vs another Random Agent on CPU

You can also give arguments to the agents or the game stats as a JSON like this,
that's in fact the args for the constructor of the classes.

    python -O reinforcement_battle.py Random DeepQlearning TicTacToe 100 --no-gpu --agent2_args='{"state_size": 9, "action_size": 9}'

Will run 100 games of TicTacToe with a Random agent vs a DeepQLearning agent on CPU,
with the passed args for the agent2 aka the DeepQLearning agent

Everything is logged into a tf_logs for the Tensorflow logs,
and in csv_logs for a custom CSV counter parts.

### Tips
You can configure things in the reinforcement_ecosytem/config.py

Agent name and Game name should correspond to their class in the code,
for example if you want to a random agent (you should use "Random"),
just strip the "Agent" or "GameRunner" part of the class.

You can see all the couple of agents avaible with "--help"

If you want to print out debug stuff remove the -O flags for python

### Avaiable game
For the moment there is :
 - TicTacToe
 - WindJammers
