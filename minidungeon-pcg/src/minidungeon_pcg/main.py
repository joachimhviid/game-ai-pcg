import gymnasium as gym
import numpy as np
import random

env = gym.make("md-pygame", render_mode="human")

treasure_collector_persona = np.array(
    [0.0, 0.9, 1.0, 0.0, 0.7, 0.6, 0.8], dtype=np.float32
)

EPISODES: int = 100
MAX_STEPS: int = 100


def main():
    for _ in range(EPISODES):
        observation, info = env.reset()
        reward_sum: float = 0.0
        
        for i in range(MAX_STEPS):
            env.render()
            observation, reward, terminated, truncated, info = env.step(treasure_collector_persona)
            reward_sum += float(reward)
            done = terminated or truncated

            if done:
                env.render()
                print(reward_sum)
                break

    env.close()
