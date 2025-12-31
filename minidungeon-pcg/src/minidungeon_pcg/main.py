import argparse
import os
from minidungeon_pcg.pcg.dungeon_generator import DungeonGenerator
from minidungeon_pcg.pcg.dungeon_generator_env import DungeonGeneratorEnv
from minidungeon_pcg.pcg.tensor_callback import CustomTensorboardCallback
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
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


def train_tilebased(args):
    model_version = args.version if args.version else 1
    model_name = f"dungeon_gen_v{model_version}"
    print(f"--- Tile-based Environment using Model v{model_version} ---")
    vec_env = make_vec_env(
        lambda: DungeonGeneratorEnv(), n_envs=4, vec_env_cls=SubprocVecEnv
    )

    if os.path.exists(f"{model_name}.zip"):
        model = PPO.load(model_name, vec_env)
        # model = PPO(
        #     "MultiInputPolicy", vec_env, verbose=0, tensorboard_log="./tensorboard/"
        # )
        # model.set_parameters(old_model.get_parameters())
    else:
        model = PPO(
            "MultiInputPolicy", vec_env, verbose=0, tensorboard_log="./tensorboard/", ent_coef=0.05
        )
    model.learn(
        total_timesteps=1_000_000,
        progress_bar=True,
        callback=CustomTensorboardCallback(),
    )
    model.save(f"{model_name}")
    print(f"Training of {model_name} complete")


def generate_tilebased(args):
    model_version = args.version if args.version else 1
    model_name = f"dungeon_gen_v{model_version}"
    difficulty_target = args.difficulty if args.difficulty else 30.0
    print(
        f"Generating dungeon with target difficulty {difficulty_target} using model {model_name}"
    )
    env = DungeonGeneratorEnv()
    obs, _ = env.reset()
    env.current_target = difficulty_target
    obs["target_reward"] = np.array([difficulty_target], dtype=np.float32)

    model = PPO.load(model_name)

    done = False

    # The agent fills the board tile by tile
    while not done:
        # Agent looks at the empty map + target and decides the next tile
        action, _ = model.predict(obs, deterministic=False)

        # Apply action
        obs, reward, done, _, _ = env.step(action)  # type: ignore

    print(DungeonGeneratorEnv.dungeon_to_str(obs["dungeon"]))
    print(obs["dungeon"])
    return obs["dungeon"]


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
        "--variant",
        type=str,
        default="tile-based",
        choices=["tile-based", "param-based"],
        help="Which generator environment version to use",
    )
    parser.add_argument(
        "--version", type=int, default=1, help="Which model version to use"
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
        "--difficulty",
        type=float,
        default=30.0,
        help="Difficulty of the level to generate.",
    )
    gen_group.add_argument(
        "--stage_prefix",
        type=str,
        default="ppo_generated",
        help="Prefix for generated level file names.",
    )

    args = parser.parse_args()

    if args.mode == "train":
        if args.variant == "tile-based":
            train_tilebased(args)
        else:
            train(args)
    elif args.mode == "generate":
        if args.variant == "tile-based":
            generate_tilebased(args)
        else:
            generate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
