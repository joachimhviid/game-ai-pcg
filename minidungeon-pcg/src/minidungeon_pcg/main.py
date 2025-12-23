import argparse
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from minidungeon_pcg.envs.generator_env import GeneratorEnv


def train(args):
    """Train a PPO model to control the generator."""
    print("--- Training Mode ---")
    env = make_vec_env(lambda: GeneratorEnv(debug=args.debug), n_envs=1)

    if os.path.exists(args.model_file) and args.continue_training:
        print(f"Loading existing model from {args.model_file} and continuing training.")
        model = PPO.load(args.model_file, env=env)
    else:
        print("Creating a new PPO model.")
        model = PPO("MlpPolicy", env, verbose=1)

    print(f"Training for {args.train_timesteps} timesteps...")
    model.learn(total_timesteps=args.train_timesteps, progress_bar=True)

    print(f"Saving model to {args.model_file}")
    model.save(args.model_file)
    print("Training complete.")


def generate(args):
    """Generate levels using a trained PPO model."""
    print("--- Generation Mode ---")
    if not os.path.exists(args.model_file):
        print(f"Error: Model file not found at {args.model_file}")
        print("Please train a model first using --mode train")
        return

    print(f"Loading model from {args.model_file}")
    model = PPO.load(args.model_file)
    env = GeneratorEnv(debug=args.debug)
    env.stage_name_prefix = args.stage_prefix

    obs, _ = env.reset()

    print(f"Generating {args.n_levels} levels with prefix '{args.stage_prefix}'...")
    for i in range(args.n_levels):
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, done, truncated, info = env.step(action)
        print(
            f"  Generated level {i+1}/{args.n_levels} -> Agent Reward: {reward:.2f}, Params: {env.generator.config}"
        )

    print("Generation complete.")
    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train or use a PPO model to generate Minidungeon levels."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "generate"],
        help="Run in 'train' or 'generate' mode.",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="generator_ppo_model.zip",
        help="Path to save/load the PPO model.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug printing.")

    # Training-specific arguments
    train_group = parser.add_argument_group("Training Arguments")
    train_group.add_argument(
        "--train_timesteps",
        type=int,
        default=1000,
        help="Number of timesteps to train the model.",
    )
    train_group.add_argument(
        "--continue_training",
        action="store_true",
        help="Continue training from an existing model file.",
    )

    # Generation-specific arguments
    gen_group = parser.add_argument_group("Generation Arguments")
    gen_group.add_argument(
        "--n_levels", type=int, default=10, help="Number of levels to generate."
    )
    gen_group.add_argument(
        "--stage_prefix",
        type=str,
        default="ppo_generated",
        help="Prefix for generated level file names.",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "generate":
        generate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
