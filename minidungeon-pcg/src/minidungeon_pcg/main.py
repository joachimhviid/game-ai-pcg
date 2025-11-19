import gymnasium as gym
import gym_md
from gym_md.envs import MdEnvBase
import random
from pathlib import Path
import sys

# env = gym.make("md-pcg", render_mode="human")

LOOP: int = 100
TRY_OUT: int = 100


def main():
    root = Path(__file__).resolve().parent
    pcg_path = root / "pcg" / "stages" / "pcg.txt"

    try:
        content = pcg_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"{pcg_path} not found", file=sys.stderr)
        return
    except Exception as e:
        print(f"Failed to read {pcg_path}: {e}", file=sys.stderr)
        return

    print(content)
    # for _ in range(TRY_OUT):
    #     observation, info = env.reset()
    #     reward_sum: float = 0.0
    #     for i in range(LOOP):
    #         env.render()
    #         action = [random.random() for _ in range(7)]
    #         observation, reward, terminated, truncated, info = env.step(action)
    #         reward_sum += float(reward)
    #         done = terminated or truncated

    #         if done:
    #             env.render()
    #             break

    #     print(reward_sum)
