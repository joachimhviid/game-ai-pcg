from typing import Any, Optional, TypeAlias
from minidungeon_pcg.envs.agent.pather import Pather
from minidungeon_pcg.envs.md_env_sim import MdEnvSim
from minidungeon_pcg.pcg.tiles import Tiles
import numpy as np
import numpy.typing as npt
import gymnasium as gym
from gymnasium import spaces

# This is a 2D array, but numpy is unable show that on a type level.
Dungeon: TypeAlias = npt.NDArray[np.str_]


class DungeonGeneratorEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    def __init__(
        self,
        map_size: tuple[int, int] = (9, 10),
    ):
        super().__init__()

        self.rows, self.cols = map_size
        self.tile_lookup = np.array(Tiles)
        self.tile_counts = np.zeros(len(Tiles), dtype=np.int32)
        self.has_start = False
        self.has_end = False
        self.pather = Pather()

        # len(Tiles) + 1 to add a submit action
        self.action_space = spaces.MultiDiscrete([self.rows, self.cols, len(Tiles) + 1])

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
                "tile_counts": spaces.Box(
                    low=0,
                    high=self.rows * self.cols,
                    shape=(len(Tiles),),
                    dtype=np.int32,
                ),
                "requirements": spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            }
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.dungeon = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.step_count = 0
        self.tile_counts = np.zeros(len(Tiles), dtype=np.int32)
        self.has_start = False
        self.has_end = False
        self.last_fitness, _ = self.calculate_dungeon_fitness()

        # Random target
        self.current_target = 30.0
        # self.current_target = np.random.uniform(5, 45)
        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "dungeon": self.dungeon,
            "target_reward": np.array([self.current_target], dtype=np.float32),
            "tile_counts": self.tile_counts,
            "requirements": np.array(
                [
                    1.0 if self.tile_counts[2] == 1 else 0.0,
                    1.0 if self.tile_counts[3] == 1 else 0.0,
                ]
            ),
        }

    def step(self, action):
        generator_reward = 0
        gen_info = {}

        terminated = False
        truncated = False

        row, col, tile = action

        if tile == len(Tiles):
            terminated = True
            self.count_dungeon_tiles()
            current_fitness, valid = self.calculate_dungeon_fitness()
            if valid:
                generator_reward += 10
        else:
            self.dungeon[row, col] = tile
            self.count_dungeon_tiles()
            current_fitness, valid = self.calculate_dungeon_fitness()

        self.step_count += 1

        generator_reward += current_fitness - self.last_fitness
        self.last_fitness = current_fitness

        truncated = self.step_count >= (self.rows * self.cols * 3)

        gen_info["tile_counts"] = self.tile_counts

        if terminated and valid:
            # build the level from ints -> tiles
            valid_dungeon = self.tile_lookup[self.dungeon]
            try:
                # agent runs through level
                sim_env = MdEnvSim(valid_dungeon)
                sim_env.reset()
                done = False
                sim_step_count = 0
                max_steps = 100  # Prevent infinite loops
                sim_timeout = not sim_step_count < max_steps
                sim_info = {}
                sim_reward = 0
                while not done and not sim_timeout:
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
                    gen_info["delta_reward"] = delta_reward
                    gen_info["target_reward"] = self.current_target
                    gen_info["sim_reward"] = sim_reward
                    gen_info["dungeon"] = self.dungeon
                    generator_reward = 50.0 - delta_reward
                else:
                    generator_reward = -50.0
            except Exception as e:
                print(f"GeneratorEnv: Error during simulation: {e}")
                generator_reward += -100

        return self._get_obs(), generator_reward, terminated, truncated, gen_info

    def calculate_dungeon_fitness(self):
        fitness = 0.0
        floors, walls, starts, exits, monsters, potions, treasures = [
            int(x) for x in self.tile_counts
        ]

        if starts == 0 or exits == 0:
            return -50, False

        # has exactly one start
        fitness += 10 if starts == 1 else -10 * (starts - 1)

        # has exactly one exit
        fitness += 10 if exits == 1 else -10 * (exits - 1)

        # start and exit has connected path
        tiled_dungeon = self.dungeon_to_str(self.dungeon)
        start_y, start_x = np.where(tiled_dungeon == Tiles.START)

        path_to_exit = self.pather.shortest_path(
            grid=list(tiled_dungeon),
            start=(start_x[0], start_y[0]),
            target_chars={Tiles.EXIT},
        )
        path_to_exit_distance = len(path_to_exit)
        if path_to_exit_distance == 0:
            return -50, False
        else:
            # start and exit not adjacent (longer path = better)
            fitness += path_to_exit_distance

        # should include all tiles
        if np.all(self.tile_counts > 0):
            fitness += 10
        else:
            missing_count = int(np.sum(self.tile_counts == 0))
            fitness -= 5 * missing_count

        # should have a certain wall ratio
        wall_ratio = walls / (self.rows * self.cols)
        if 0.20 <= wall_ratio <= 0.60:
            fitness += 15
        else:
            fitness -= abs(0.575 - wall_ratio) * 30

        return int(fitness), True

    def count_dungeon_tiles(self):
        for i, tile in enumerate(Tiles):
            y, x = np.where(self.dungeon == i)
            self.tile_counts[i] = len(y)

        return self.tile_counts

    @staticmethod
    def dungeon_to_str(dungeon):
        return np.array(Tiles)[dungeon]
