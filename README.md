# Reinforcement learning project for 5IBD ESGI 2018-2019
This project aim to teach us reinforcement learning by implementing the popular RL models.


## TODO
- Better docstrings (understand '???' comments)
- Better type hints (find type of 'Any' type hints)
- Transform get methods in python property
- Refactor GameRunner random_rollout_run function to be more generic 
- Add application logs
- Tests everything
- Profile and rework heavy functions
  - WindJammersGameState.compute_current_score_and_end_game_more_efficient
  - WindJammersGameState.frisbee_hitplayer1
  - WindJammersGameState.frisbee_hitplayer2
- Comments Windjamers game
- Clean game_runner.py and reinforcement_battle.py


## How to use ?
Be careful you need Python 3.6 or higher
You have to use the reinforcement_battle.py in command line.

Example :

    python -O reinforcement_battle.py Random Random TicTacToe 100 --no-gpu

Will run 100 games of TicTacToe with a Random agent vs another Random Agent on CPU

You can also give arguments to the agents or the game stats as a JSON like this,
that's in fact the args for the constructor of the classes.

    python -O reinforcement_battle.py Random DeepQlearning TicTacToe 100 --no-gpu \ 
    --agent2_args='{"state_size": 9, "action_size": 9}'

Will run 100 games of TicTacToe with a Random agent vs a DeepQLearning agent on CPU,
with the passed args for the agent2 aka the DeepQLearning agent

Everything is logged into a tf_logs for the Tensorflow logs,
and in csv_logs for his custom CSV counter parts.

Basically pass the class parameter as JSON in the --agentX_args,
where X is the number of the agent based on his order in the command


### Tips
You can configure things in the reinforcement_ecosytem/config.py

Agent name and Game name should correspond to their class in the code,
for example if you want to a random agent (you should use "Random"),
just strip the "Agent" or "GameRunner" part of the class.

If you want to print out debug stuff remove the -O flags for python


### Available agents
Tested on TicTacToe (WindJammers should work too)
- Random
- RandomRollout
- ReinforceClassic
- ReinforceClassicWithMultipleTrajectories
- MOISMCTSWithValueNetwork
- MOISMCTSWithRandomRollouts 
- MOISMCTSWithRandomRolloutsExpertThenApprentice
- TabularQLearning 
- DeepQLearning


### Arguments for the agents
Arguments can be passed to the agent class constructor dynamically via a JSON string
with --agentX_args where X is the number of the agent based on his position on the command

    python reinforcement_battle.py agent1 agent2 ...

Here is the list of all the mandatory argument to give, depending on the agent class called and the game

Of course you can pass optionnal arguments in the same fashion, please refer to the class documentation for more info
- TicTacToe
    - Random
        - No need to inquire anything here
    - RandomRollout
        - (int) num_rollouts_per_available_action : whatever int you want here
        - (str) runner : "TicTacToe"
    - MOISMCTSWithRandomRollouts
        - (int) iteration_count : whatever int you want here
        - (str) runner : "TicTacToe"
    - MOISMCTSWithRandomRolloutsExpertThenApprentice
        - (int) iteration_count : whatever int you want here
        - (str) runner : "TicTacToe"
        - (int) state_size : 9
        - (int) action_size : 9
    - MOISMCTSWithValueNetwork
        - (int) iteration_count : whatever int you want here
        - (int) state_size : 9
        - (int) num_players : 2
    - ReinforceClassic
        - (int) state_size : 9
        - (int) action_size : 9
    - ReinforceClassicWithMultipleTrajectories
        - (int) state_size : 9
        - (int) action_size : 9
    - PPOWithMultipleTrajectoriesMultiOutputs
        - (int) state_size : 9
        - (int) action_size : 9
    - DeepQLearning
        - (int) input_size : 9
        - (int) action_size : 9
    - TabularQLearning
         - No need to inquire anything here
- WindJammers
    - Random
        - No need to inquire anything here
    - RandomRollout
        - (int) num_rollouts_per_available_action : whatever int you want here
        - (str) runner : "WindJammers"
    - MOISMCTSWithRandomRollouts
        - (int) iteration_count : whatever int you want here
        - (str) runner : "WindJammers"
    - MOISMCTSWithRandomRolloutsExpertThenApprentice
        - (int) iteration_count : whatever int you want here
        - (str) runner : "WindJammers"
        - (int) state_size : 8
        - (int) action_size : 12
    - MOISMCTSWithValueNetwork
        - (int) iteration_count : whatever int you want here
        - (int) state_size : 8
        - (int) num_players : 2
    - ReinforceClassic
        - (int) state_size : 8
        - (int) action_size : 12
    - ReinforceClassicWithMultipleTrajectories
        - (int) state_size : 8
        - (int) action_size : 12
    - PPOWithMultipleTrajectoriesMultiOutputs
        - (int) state_size : 8
        - (int) action_size : 12
    - DeepQLearning
        - (int) input_size : 8
        - (int) action_size : 12
    - TabularQLearning
         - No need to inquire anything here

### Examples
Here are some example for the agents

    # Will run 100 games of TicTacToe Random VS RandomRollout 
    python -O reinforcement_battle.py Random RandomRollout TicTacToe 100 --no-gpu \ 
    --agent2_args='{"num_rollouts_per_available_action": 9, "runner": "TicTacToe" }'
    
    # Will run 10000 Game of TicTacToe MOISMCTSWithRandomRolloutsExpertThenApprentice VS TabularQLearning 
    python -O reinforcement_battle.py MOISMCTSWithRandomRolloutsExpertThenApprentice TabularQLearning TicTacToe \
    10000 --no-gpu --agent1_args='{"iteration_count": 9, "runner": "TicTacToe", "state_size": 9, "action_size": 9}' \
    --agent2_args='{"input_size": 9, "action_size": 9}'


### Available game
For the moment there is :
 - TicTacToe
 - WindJammers
 