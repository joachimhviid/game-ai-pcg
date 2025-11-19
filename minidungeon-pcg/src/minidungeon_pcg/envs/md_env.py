from typing import Any
import gymnasium as gym
import numpy as np


class MdEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["human"]}

    def __init__(self, stage_name: str):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,))
        # self.observation_space = gym.spaces.Box(
        #     low=0, high=self.setting.DISTANCE_INF, shape=(8,), dtype=np.int32
        # )

    def step(self, action):
        return super().step(action)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        return super().reset(seed=seed, options=options)

    def render(self):
        return super().render()

    def close(self):
        return super().close()
