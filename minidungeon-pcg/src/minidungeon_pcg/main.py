import gymnasium as gym

env = gym.make("md-pygame", render_mode="human")

EPISODES: int = 100
MAX_STEPS: int = 100


def main():
    for _ in range(EPISODES):
        observation, info = env.reset()
        reward_sum: float = 0.0

        for i in range(MAX_STEPS):
            env.render()
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            reward_sum += float(reward)
            done = terminated or truncated

            if done:
                env.render()
                print(reward_sum)
                break

    env.close()
