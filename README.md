# acrobot
RL agent for Gym acrobot environment

## Installation

Install the project environment
```
conda env create -f env.yml
conda activate acrobot
```

## Training

Train the agent with
```
python train.py
```

The agent pytorch model will be saved in a file called `state.pt`. 
You can change this name and other parameter in the file `train.py`.

## Simulation

Simulate the environment and the agent with
```
python test.py
```
