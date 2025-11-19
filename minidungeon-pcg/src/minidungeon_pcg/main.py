import gymnasium as gym
import random

env = gym.make("md-pcg", render_mode="human")

LOOP: int = 100
TRY_OUT: int = 100


def main():
    for _ in range(TRY_OUT):
        observation, info = env.reset()
        reward_sum: float = 0.0
        for i in range(LOOP):
            env.render()
            action = [random.random() for _ in range(7)]
            observation, reward, terminated, truncated, info = env.step(action)
            reward_sum += float(reward)
            done = terminated or truncated

            if done:
                env.render()
                break

        print(reward_sum)
