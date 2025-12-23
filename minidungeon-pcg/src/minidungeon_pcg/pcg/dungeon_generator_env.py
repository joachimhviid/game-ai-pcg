from typing import Any, Optional, TypeAlias
from minidungeon_pcg.envs.md_env_sim import MdEnvSim
from minidungeon_pcg.pcg.tiles import Tiles
import numpy as np
import numpy.typing as npt
import gymnasium as gym
from gymnasium import spaces

# This is a 2D array, but numpy is unable show that on a type level.
Dungeon: TypeAlias = npt.NDArray[np.str_]


class DungeonGeneratorEnv(gym.Env[dict[str, np.ndarray], int]):
    def __init__(
        self,
        map_size: tuple[int, int] = (10, 10),
    ):
        super().__init__()
        
        self.rows, self.cols = map_size
        self.tile_lookup = np.array(Tiles)

        # Actions are the tiles to place
        self.action_space = spaces.Discrete(len(Tiles))

        # Observation is the dungeon map and the target reward
        self.observation_space = spaces.Dict(
            {
                "dungeon": spaces.Box(
                    low=0,
                    high=len(Tiles) - 1,
                    shape=(self.rows, self.cols),
                    dtype=np.int32,
                ),
                # TODO: high and low probably needs to be adjusted
                "target_reward": spaces.Box(
                    low=0, high=50, shape=(1,), dtype=np.float32
                ),
            }
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.dungeon = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.step_count = 0

        # Random target
        self.current_target = np.random.uniform(5, 45)
        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "dungeon": self.dungeon,
            "target_reward": np.array([self.current_target], dtype=np.float32),
        }

    def step(self, action: int):
        row = self.step_count // self.cols
        col = self.step_count % self.cols
        self.dungeon[row, col] = action
        self.step_count += 1

        terminated = self.step_count >= (self.rows * self.cols)
        truncated = False

        generator_reward = 0
        if terminated:
            # build the level from ints -> tiles
            valid_dungeon = self.tile_lookup[self.dungeon]
            try:
                # agent runs through level
                sim_env = MdEnvSim(valid_dungeon)
                sim_env.reset()
                done = False
                sim_step_count = 0
                max_steps = 100  # Prevent infinite loops
                sim_info = {}
                sim_reward = 0
                while not done and sim_step_count < max_steps:
                    # The MdTreasureAgent inside env ignores this action
                    sim_action = np.zeros(sim_env.action_space.shape)  # type: ignore
                    obs, reward, sim_terminated, sim_truncated, info = sim_env.step(
                        sim_action
                    )
                    sim_info = info
                    sim_reward += reward
                    done = sim_terminated or sim_truncated
                    sim_step_count += 1

                sim_env.close()

                if sim_info["solvable"]:
                    delta_reward = abs(sim_reward - self.current_target)
                    generator_reward = 50.0 - delta_reward
                else:
                    generator_reward = -50.0

            except Exception as e:
                print(f"GeneratorEnv: Error during simulation: {e}")
                return self._get_obs(), -100.0, True, False, {}

        return self._get_obs(), generator_reward, terminated, truncated, {}
    
    @staticmethod
    def dungeon_to_str(dungeon):
        return np.array(Tiles)[dungeon]
