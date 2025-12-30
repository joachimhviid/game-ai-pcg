from typing import Optional, TypeAlias
from minidungeon_pcg.pcg.tiles import Tiles
import numpy as np
import numpy.typing as npt

# This is a 2D array, but numpy is unable show that on a type level.
Dungeon: TypeAlias = npt.NDArray[np.str_]


class DungeonGenerator:
    def __init__(
        self,
        target_reward: int,
        population_size: int,
        generation_size: int,
        map_size: tuple[int, int],
    ):
        self.target_reward = target_reward
        self.population_size = population_size
        self.generation_size = generation_size
        self.rows, self.cols = map_size

    def generate_dungeon(self):
        population = self.initialize_population()

        best_dungeon: Optional[Dungeon] = None
        best_fitness: float = float("-inf")

    def save_dungeon(self):
        pass

    def initialize_population(self):
        population: list[Dungeon] = []
        # Start/Exit will be static when initializing population
        dynamic_tiles: list[Tiles] = [
            Tiles.WALL,
            Tiles.FLOOR,
            Tiles.TREASURE,
            Tiles.MONSTER,
            Tiles.POTION,
        ]

        for _ in range(self.population_size):
            level: npt.NDArray[np.str_] = np.random.choice(
                a=dynamic_tiles,
                size=(self.rows, self.cols),
                p=[0.3, 0.4, 0.1, 0.1, 0.1],
            )
            level[0][0] = Tiles.START
            level[self.rows - 1][self.cols - 1] = Tiles.EXIT
            population.append(level)

        return population

    def calculate_fitness(self):
        pass

    def select_parents(self):
        pass

    def crossover(self):
        pass

    def mutate(self):
        pass
