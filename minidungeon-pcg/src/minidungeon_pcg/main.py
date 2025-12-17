import gymnasium as gym
import argparse
import numpy as np
from typing import List
from minidungeon_pcg.pcg.generator import Generator

# Default settings for the training loop
DEFAULT_BATCH_SIZE: int = 10
DEFAULT_PLAYS_PER_LEVEL: int = 10
DEFAULT_MAX_STEPS: int = 200
DEFAULT_MODEL_PATH: str = "generator_model.json"


def main():
    parser = argparse.ArgumentParser(
        description="Run the Minidungeon PCG environment with an agent and adaptive generator."
    )

    # Configuration arguments
    parser.add_argument(
        "--render_mode",
        type=str,
        default="human",
        help="The rendering mode for the environment.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of levels to generate and play in a batch.",
    )
    parser.add_argument(
        "--plays_per_level",
        type=int,
        default=DEFAULT_PLAYS_PER_LEVEL,
        help="Number of times the agent plays each level.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Max steps per episode.",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to save/load the generator model config.",
    )

    # Use parse_known_args to avoid conflicts with stage_name argument from __init__.py
    args, unknown = parser.parse_known_args()

    # --- 1. Initialize and Load Generator ---
    print("=" * 60)
    print("Initializing Generator...")
    generator = Generator()
    generator.load_model(args.model_file)
    print(
        f"Generator loaded. Current difficulty: {generator.config.get('difficulty_level', 1)}"
    )
    print("=" * 60)

    # --- 2. Generate a Batch of Levels ---
    print(f"Generating a batch of {args.batch_size} levels...")
    level_names = generator.generate_batch(
        batch_size=args.batch_size, stage_name_prefix="adaptive_run"
    )
    if not level_names:
        print("Level generation failed. Exiting.")
        return
    print("=" * 60)

    batch_rewards: List[float] = []
    batch_wins: List[int] = []

    # --- 3. Run Agent on the Batch of Levels ---
    for i, level_name in enumerate(level_names):
        print(f"\n--- Playing Level {i+1}/{len(level_names)}: {level_name} ---")

        # Create a new environment for each level dynamically
        env = gym.make("md-pygame", render_mode=args.render_mode, stage_name=level_name)

        for episode in range(args.plays_per_level):
            observation, info = env.reset()
            reward_sum: float = 0.0
            done = False

            for step in range(args.max_steps):
                if args.render_mode == "human":
                    env.render()
                action = np.zeros(env.action_space.shape, dtype=float) # type: ignore
                observation, reward, terminated, truncated, info = env.step(action)
                reward_sum += float(reward)
                done = terminated or truncated

                if done:
                    # Win detection based on positive reward at termination
                    is_win = terminated and float(reward) > 0
                    batch_wins.append(1 if is_win else 0)
                    break

            batch_rewards.append(reward_sum)
            if not done:  # Episode was truncated (timed out)
                batch_wins.append(0)

            print(
                f"  Episode {episode + 1}/{args.plays_per_level}: Total Reward = {reward_sum:.2f}"
            )

        env.close()

    # --- 4. Update and Save Generator ---
    print("\n" + "=" * 60)
    print("Batch finished. Analyzing performance...")

    if batch_rewards:
        avg_reward = np.mean(batch_rewards, dtype=float)
        win_rate = np.mean(batch_wins, dtype=float)

        print(f"Overall Avg Reward: {avg_reward:.2f}")
        print(f"Overall Win Rate:   {win_rate:.2%}")

        # Adapt difficulty based on agent results
        generator.update_difficulty(avg_reward=avg_reward, win_rate=win_rate)
        generator.save_model(args.model_file)
    else:
        print("No episodes were run. Generator not updated.")

    print("=" * 60)


if __name__ == "__main__":
    main()
