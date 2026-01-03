# MiniDungeons PCG

## Prerequisites

1. Python 3.13 installed
2. Poetry installed (e.g. `pipx install poetry`)

## Run Project

1. Activate the virtual environment
    - Bash/Zsh/Csh: `eval $(poetry env activate)`
    - Powershell: `Invoke-Expression (poetry env activate)`
2. `poetry install`
3. `poetry run play <name-of-stage>` (You must generate a stage first, see below)

## Information

T = CHEST  
. = WALKING TILE  
\# = NON-WALKING TILE  
M = MONSTER  
S = SPAWN  
P = POTION  
E = EXIT/STAIRS

## Running levels with agent visualized

`poetry run play --all`
`poetry run play <name-of-stage>`

## Running RL generator
### Train new generator model
`poetry run start --mode train --train_timesteps 10000`

### Continue training existing model
`poetry run start --mode train --version <version>`

### Generate levels using trained model and optionally save/play after
`poetry run start --mode generate --version <version, 19 is latest> --difficulty <5-50, default 30> --post <"none", "save", "play", default "none">`

### Running benchmark
`poetry run start --mode benchmark --version <version, 19 is latest> --difficulty <5-50, default 30> --n_levels <number-of-levels-to-generate>`


## Running GA generator

### Generate n_levels using GA
`poetry run start --mode generate --variant param-based --n_levels 10`

## Inspecting Tensorboard logs

`tensorboard --logdir="./tensorboard/"`
