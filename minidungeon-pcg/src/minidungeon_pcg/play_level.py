import argparse
import time
import numpy as np
from minidungeon_pcg.envs.md_env import MdEnv


def play(stage_name: str):
    """
    Plays a level with rendering enabled.
    """
    print(f"--- Playing Level: {stage_name} ---")
    env = None
    try:
        env = MdEnv(stage_name=stage_name, render_mode="human")

        obs, info = env.reset()
        env.render()
        time.sleep(1)  # Pause to see initial state

        done = False
        total_reward = 0
        step_count = 0
        max_steps = 100  # Safety break

        while not done and step_count < max_steps:
            action = np.zeros(env.action_space.shape)  # type: ignore
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            env.render()

            step_count += 1
            time.sleep(0.1)  # Slow down for visibility

        print(f"--- Playback Complete ---")
        print(f"Total reward: {total_reward}")
        print(f"Steps: {step_count}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print(
            "Please make sure the stage name is correct and the file exists in 'minidungeon-pcg/src/minidungeon_pcg/pcg/stages/'."
        )
        print("Example: 'ppo_generated_0' (without the .txt extension)")
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Play back a generated Minidungeon level with visual rendering."
    )
    parser.add_argument(
        "stage_name",
        nargs="?",
        default=None,
        type=str,
        help="The name of the stage file to play (e.g., 'ppo_generated_0').",
    )
    parser.add_argument(
        "--n_levels",
        type=int,
        default=0,
        help="Play n levels, starting from ppo_generated_0.",
    )
    parser.add_argument(
        "--all", action="store_true", help="Play all 15 generated levels."
    )
    args = parser.parse_args()

    if args.all:
        for i in range(15):
            stage_name = f"ppo_generated_{i}"
            play(stage_name)
    elif args.n_levels > 0:
        for i in range(args.n_levels):
            stage_name = f"ppo_generated_{i}"
            play(stage_name)
    elif args.stage_name:
        play(args.stage_name)
    else:
        print("No stage specified. Use a stage name, --n_levels, or --all.")


if __name__ == "__main__":
    main()
