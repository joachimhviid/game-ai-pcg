# MiniDungeons PCG

## Prerequisites

1. Python 3.13 installed
2. Poetry installed (e.g. `pipx install poetry`)

## Run Project

1. Activate the virtual environment
    - Bash/Zsh/Csh: `eval $(poetry env activate)`
    - Powershell: `Invoke-Expression (poetry env activate)`
2. `poetry install`
3. `poetry run play <name-of-stage>` (Included stage is "pcg")

## Information

T = CHEST  
. = WALKING TILE  
\# = NON-WALKING TILE  
M = MONSTER  
S = SPAWN  
P = POTION  
E = EXIT/STAIRS

## Running PPO
# Train new generator model
poetry run start --mode train --train_timesteps 10000

# Continue training existing model
poetry run start --mode train --continue_training

# Generate n levels using trained model
poetry run start --mode generate --n_levels 15

## Running levels with agent visualized
poetry run play --all
poetry run play ppo_generated_n 
