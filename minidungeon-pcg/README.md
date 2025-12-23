## Prerequisites
1. Install poetry

## Run Project
1. cd /game-ai-pcg/minidungeon-pcg
2. install poetry
3. poetry run start

## Run game
poetry run start ga_generated --env md-pygame --batch_size 10 --plays_per_level 10


## Information
T = CHEST
. = WALKING TILE
"#" = NON-WALKING TILE
M = MONSTER
S = SPAWN
P = POTION
E = EXIT/STAIRS

## Setup
poetry install

## Running PPO
# Train new generator model
poetry run python -m minidungeon_pcg.main --mode train --train_timesteps 10000

# Continue training existing model
poetry run python -m minidungeon_pcg.main --mode train --continue_training

# Generate n levels using trained model
poetry run python -m minidungeon_pcg.main --mode generate --n_levels 15

## Running PPO levels with agent visualized
poetry run python play_level.py --all
poetry run python play_level.py ppo_generated_n 
