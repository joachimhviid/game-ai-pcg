import gymnasium as gym
import gym_md
import random

env = gym.make('md-test-v0', render_mode='human')

LOOP: int = 10
TRY_OUT: int = 1

def main():
    print('running main!!!')
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