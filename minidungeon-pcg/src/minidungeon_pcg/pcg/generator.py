import random
import copy
import json
import os
from typing import List, Tuple, Dict
from collections import deque
from os import path


class Generator:
    """
    Creates stage using Genetic Algorithm and saves to /pcg/stages
    Supports batch generation and adaptive difficulty.
    """

    def __init__(
        self,
        width: int = 9,
        height: int = 9,
        population_size: int = 150,
        generations: int = 300,
        mutation_rate: float = 0.15,
        elite_size: int = 10,
    ) -> None:
        self.width = width
        self.height = height
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size

        # Tile types
        self.WALL = "#"
        self.FLOOR = "."
        self.START = "S"
        self.EXIT = "E"
        self.MONSTER = "M"
        self.POTION = "P"
        self.TREASURE = "T"

        # Adaptive Configuration (Saved/Loaded)
        self.config = {
            "min_path_length": 8,
            "target_monster_count": 3,
            "target_potion_count": 1,
            "target_treasure_count": 3,
            "difficulty_level": 1,
        }

        self.update_internal_params()

    def update_internal_params(self):
        """Sync internal class vars with config dict"""
        self.min_path_length = self.config["min_path_length"]
        self.target_monster_count = self.config["target_monster_count"]
        self.target_potion_count = self.config["target_potion_count"]
        self.target_treasure_count = self.config["target_treasure_count"]

    def generate_batch(
        self, batch_size: int = 10, stage_name_prefix: str = "generated"
    ) -> List[str]:
        """
        Generates a batch of levels and returns their names.
        """
        generated_files = []
        print(f"--- Starting Batch Generation (Size: {batch_size}) ---")

        # We can optionally keep the population between generations to "evolve" the batch
        # For now, we initialize fresh for diversity, but you could persist self.population

        for i in range(batch_size):
            name = f"{stage_name_prefix}_{i}"
            print(f"Generating level {i+1}/{batch_size}: {name}")
            try:
                self.generate_dungeon(stage_name=name)
                generated_files.append(name)
            except Exception as e:
                print(f"Failed to generate level {i}: {e}")

        return generated_files

    def update_difficulty(self, avg_reward: float, win_rate: float):
        """
        Adaptive Difficulty Scaling:
        - If agent is winning easily (High Reward/Win Rate) -> Increase Difficulty
        - If agent is struggling (Low Reward/Win Rate) -> Decrease Difficulty
        """
        print(
            f"Updating Generator based on: Avg Reward={avg_reward:.2f}, Win Rate={win_rate:.2f}"
        )

        # Simple heuristic thresholds (Adjust these based on your reward scale)
        # Assuming Max Reward ~20-30 per level

        if win_rate > 0.8:  # Too Easy
            print(">> Increasing Difficulty")
            self.config["difficulty_level"] += 1
            self.config["target_monster_count"] += 1
            self.config["target_potion_count"] = max(
                0, self.config["target_potion_count"] - 1
            )
            self.config["min_path_length"] += 2

        elif win_rate < 0.2:  # Too Hard
            print(">> Decreasing Difficulty")
            self.config["difficulty_level"] = max(
                1, self.config["difficulty_level"] - 1
            )
            self.config["target_monster_count"] = max(
                1, self.config["target_monster_count"] - 1
            )
            self.config["target_potion_count"] += 1
            self.config["min_path_length"] = max(5, self.config["min_path_length"] - 2)

        else:
            print(">> Difficulty maintained")

        # Cap values to prevent breaking the generator
        self.config["target_monster_count"] = min(
            10, self.config["target_monster_count"]
        )
        self.config["target_potion_count"] = min(5, self.config["target_potion_count"])

        self.update_internal_params()

    def save_model(self, filename: str = "generator_model.json"):
        """Save the current generator configuration"""
        file_dir = path.dirname(__file__)
        filepath = path.join(file_dir, filename)
        with open(filepath, "w") as f:
            json.dump(self.config, f, indent=4)
        print(f"Generator model saved to {filepath}")

    def load_model(self, filename: str = "generator_model.json"):
        """Load generator configuration"""
        file_dir = path.dirname(__file__)
        filepath = path.join(file_dir, filename)
        if path.exists(filepath):
            with open(filepath, "r") as f:
                self.config = json.load(f)
            self.update_internal_params()
            print(f"Generator model loaded from {filepath}")
        else:
            print("No saved model found, using defaults.")

    # ... [Keep your existing generate_dungeon, initialize_population, etc. methods here] ...
    # Ensure generate_dungeon uses self.target_monster_count etc. (which it already does)

    # Below is the rest of your original code (abbreviated for the response)
    # Be sure to include the full original class implementation here.

    def generate_dungeon(self, stage_name: str = "generated") -> List[List[str]]:
        # ... (Your original implementation) ...
        # Make sure to remove the hardcoded print at the end or change it to debug
        # COPY PASTE YOUR ORIGINAL generate_dungeon METHOD AND HELPERS HERE

        # [Use the exact code you provided in the upload for the rest of the class]
        print(f"Initializing GA with population size {self.population_size}...")
        population = self.initialize_population()

        best_fitness = float("-inf")
        best_dungeon = None
        generations_without_improvement = 0

        for generation in range(self.generations):
            fitnesses = [self.calculate_fitness(dungeon) for dungeon in population]
            max_fitness_idx = fitnesses.index(max(fitnesses))

            if fitnesses[max_fitness_idx] > best_fitness:
                best_fitness = fitnesses[max_fitness_idx]
                best_dungeon = copy.deepcopy(population[max_fitness_idx])
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            new_population = []
            elite_indices = sorted(
                range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True
            )[: self.elite_size]
            for idx in elite_indices:
                new_population.append(copy.deepcopy(population[idx]))

            while len(new_population) < self.population_size:
                parent1 = self.selection(population, fitnesses)
                parent2 = self.selection(population, fitnesses)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            population = new_population

        if best_dungeon:
            self.save_dungeon(best_dungeon, stage_name)
            return best_dungeon
        else:
            raise Exception("Failed to generate a valid dungeon")

    # ... Include all other helper methods (initialize_population, etc.) from your upload ...
    # ... I am omitting them here for brevity, but they are required ...

    def initialize_population(self) -> List[List[List[str]]]:
        """Create initial random population of dungeons"""
        population = []
        for i in range(self.population_size):
            if i < int(self.population_size * 0.7):
                dungeon = self.create_structured_dungeon()
            else:
                dungeon = self.create_random_dungeon()
            population.append(dungeon)
        return population

    def create_random_dungeon(self) -> List[List[str]]:
        dungeon = [[self.FLOOR for _ in range(self.width)] for _ in range(self.height)]
        wall_count = random.randint(
            int(self.width * self.height * 0.5), int(self.width * self.height * 0.6)
        )
        for _ in range(wall_count):
            x, y = random.randint(0, self.height - 1), random.randint(0, self.width - 1)
            dungeon[x][y] = self.WALL

        start_positions = [(0, 0), (0, self.width - 1), (self.height - 1, 0)]
        exit_positions = [
            (self.height - 1, self.width - 1),
            (self.height // 2, self.width - 1),
        ]

        start_x, start_y = random.choice(start_positions)
        exit_x, exit_y = random.choice(exit_positions)

        dungeon[start_x][start_y] = self.START
        dungeon[exit_x][exit_y] = self.EXIT

        for _ in range(self.target_monster_count):
            x, y = self.find_empty_position(dungeon)
            if x is not None:
                dungeon[x][y] = self.MONSTER

        for _ in range(self.target_potion_count):
            x, y = self.find_empty_position(dungeon)
            if x is not None:
                dungeon[x][y] = self.POTION

        for _ in range(self.target_treasure_count):
            x, y = self.find_empty_position(dungeon)
            if x is not None:
                dungeon[x][y] = self.TREASURE

        return dungeon

    def create_structured_dungeon(self) -> List[List[str]]:
        dungeon = [[self.WALL for _ in range(self.width)] for _ in range(self.height)]
        num_corridors = random.randint(3, 5)
        for _ in range(num_corridors):
            x, y = random.randint(1, self.height - 2), random.randint(1, self.width - 2)
            corridor_length = random.randint(8, 15)

            for step in range(corridor_length):
                dungeon[x][y] = self.FLOOR
                if random.random() < 0.3:
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.height and 0 <= ny < self.width:
                            dungeon[nx][ny] = self.FLOOR
                direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                x = max(1, min(self.height - 2, x + direction[0]))
                y = max(1, min(self.width - 2, y + direction[1]))

        num_rooms = random.randint(2, 4)
        for _ in range(num_rooms):
            room_x = random.randint(1, self.height - 4)
            room_y = random.randint(1, self.width - 4)
            room_w = random.randint(2, 3)
            room_h = random.randint(2, 3)

            for i in range(room_h):
                for j in range(room_w):
                    if room_x + i < self.height and room_y + j < self.width:
                        dungeon[room_x + i][room_y + j] = self.FLOOR

        start_positions = [(1, 1), (1, self.width - 2), (self.height - 2, 1)]
        exit_positions = [
            (self.height - 2, self.width - 2),
            (self.height // 2, self.width - 2),
        ]

        start_x, start_y = random.choice(start_positions)
        exit_x, exit_y = random.choice(exit_positions)

        dungeon[start_x][start_y] = self.START
        dungeon[exit_x][exit_y] = self.EXIT

        if not self.calculate_path_length(
            dungeon, (start_x, start_y), (exit_x, exit_y)
        )[1]:
            self.carve_path(dungeon, (start_x, start_y), (exit_x, exit_y))

        for _ in range(self.target_monster_count):
            x, y = self.find_empty_position(dungeon)
            if x is not None:
                dungeon[x][y] = self.MONSTER

        for _ in range(self.target_potion_count):
            x, y = self.find_empty_position(dungeon)
            if x is not None:
                dungeon[x][y] = self.POTION

        for _ in range(self.target_treasure_count):
            x, y = self.find_empty_position(dungeon)
            if x is not None:
                dungeon[x][y] = self.TREASURE

        return dungeon

    def find_empty_position(self, dungeon: List[List[str]]) -> Tuple[int, int]:
        attempts = 0
        while attempts < 100:
            x, y = random.randint(0, self.height - 1), random.randint(0, self.width - 1)
            if dungeon[x][y] == self.FLOOR:
                return x, y
            attempts += 1
        return None, None

    def calculate_fitness(self, dungeon: List[List[str]]) -> float:
        fitness = 0.0
        start_pos = self.find_tile(dungeon, self.START)
        exit_pos = self.find_tile(dungeon, self.EXIT)

        if not start_pos or not exit_pos:
            return -1000.0

        path_length, path_exists = self.calculate_path_length(
            dungeon, start_pos, exit_pos
        )
        if not path_exists:
            fitness -= 500
        else:
            if path_length >= self.min_path_length:
                fitness += min(30, path_length * 2)
            else:
                fitness -= (self.min_path_length - path_length) * 5

        reachable_tiles = self.count_reachable_tiles(dungeon, start_pos)
        total_floor_tiles = self.count_floor_tiles(dungeon)
        if total_floor_tiles > 0:
            connectivity_ratio = reachable_tiles / total_floor_tiles
            fitness += connectivity_ratio * 25

        monster_positions = self.find_all_tiles(dungeon, self.MONSTER)
        if len(monster_positions) > 0:
            min_distance = self.calculate_min_distance_between_entities(
                monster_positions
            )
            fitness += min(20, min_distance * 4)

            if path_exists:
                monsters_on_path = self.count_entities_near_path(
                    dungeon, start_pos, exit_pos, self.MONSTER, distance=2
                )
                fitness += monsters_on_path * 3

        potion_count = len(self.find_all_tiles(dungeon, self.POTION))
        treasure_count = len(self.find_all_tiles(dungeon, self.TREASURE))
        monster_count = len(self.find_all_tiles(dungeon, self.MONSTER))

        if monster_count > self.target_monster_count + 2:
            fitness -= (monster_count - self.target_monster_count - 2) * 50

        fitness += max(0, 10 - abs(potion_count - self.target_potion_count) * 3)
        fitness += max(0, 5 - abs(treasure_count - self.target_treasure_count) * 2)
        fitness += max(0, 10 - abs(monster_count - self.target_monster_count) * 5)

        dead_end_count = self.count_dead_ends(dungeon)
        fitness -= dead_end_count * 2

        wall_count = sum(row.count(self.WALL) for row in dungeon)
        total_tiles = self.width * self.height
        wall_ratio = wall_count / total_tiles

        if 0.50 <= wall_ratio <= 0.65:
            fitness += 15
        else:
            fitness -= abs(0.575 - wall_ratio) * 30

        if path_exists and path_length > self.min_path_length + 5:
            fitness += min(10, (path_length - self.min_path_length) * 1.5)

        return fitness

    def carve_path(
        self, dungeon: List[List[str]], start: Tuple[int, int], end: Tuple[int, int]
    ) -> None:
        x, y = start
        target_x, target_y = end

        while (x, y) != (target_x, target_y):
            dungeon[x][y] = self.FLOOR
            if x < target_x:
                x += 1
            elif x > target_x:
                x -= 1
            elif y < target_y:
                y += 1
            elif y > target_y:
                y -= 1
            if random.random() < 0.2:
                direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                nx, ny = x + direction[0], y + direction[1]
                if 0 <= nx < self.height and 0 <= ny < self.width:
                    x, y = nx, ny

    def calculate_path_length(
        self, dungeon: List[List[str]], start: Tuple[int, int], end: Tuple[int, int]
    ) -> Tuple[int, bool]:
        queue = deque([(start, 0)])
        visited = {start}
        while queue:
            (x, y), dist = queue.popleft()
            if (x, y) == end:
                return dist, True
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < self.height
                    and 0 <= ny < self.width
                    and (nx, ny) not in visited
                    and dungeon[nx][ny] != self.WALL
                ):
                    visited.add((nx, ny))
                    queue.append(((nx, ny), dist + 1))
        return 0, False

    def count_reachable_tiles(
        self, dungeon: List[List[str]], start: Tuple[int, int]
    ) -> int:
        queue = deque([start])
        visited = {start}
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < self.height
                    and 0 <= ny < self.width
                    and (nx, ny) not in visited
                    and dungeon[nx][ny] != self.WALL
                ):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return len(visited)

    def count_floor_tiles(self, dungeon: List[List[str]]) -> int:
        count = 0
        for row in dungeon:
            for tile in row:
                if tile != self.WALL:
                    count += 1
        return count

    def count_dead_ends(self, dungeon: List[List[str]]) -> int:
        dead_ends = 0
        for i in range(self.height):
            for j in range(self.width):
                if dungeon[i][j] != self.WALL:
                    adjacent_walls = 0
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = i + dx, j + dy
                        if (
                            nx < 0
                            or nx >= self.height
                            or ny < 0
                            or ny >= self.width
                            or dungeon[nx][ny] == self.WALL
                        ):
                            adjacent_walls += 1
                    if adjacent_walls >= 3:
                        dead_ends += 1
        return dead_ends

    def find_tile(self, dungeon: List[List[str]], tile_type: str) -> Tuple[int, int]:
        for i in range(self.height):
            for j in range(self.width):
                if dungeon[i][j] == tile_type:
                    return (i, j)
        return None

    def find_all_tiles(
        self, dungeon: List[List[str]], tile_type: str
    ) -> List[Tuple[int, int]]:
        positions = []
        for i in range(self.height):
            for j in range(self.width):
                if dungeon[i][j] == tile_type:
                    positions.append((i, j))
        return positions

    def calculate_min_distance_between_entities(
        self, positions: List[Tuple[int, int]]
    ) -> float:
        if len(positions) < 2:
            return 0
        min_dist = float("inf")
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = abs(positions[i][0] - positions[j][0]) + abs(
                    positions[i][1] - positions[j][1]
                )
                min_dist = min(min_dist, dist)
        return min_dist

    def count_entities_near_path(
        self,
        dungeon: List[List[str]],
        start: Tuple[int, int],
        end: Tuple[int, int],
        entity_type: str,
        distance: int = 2,
    ) -> int:
        path_tiles = self.get_path_tiles(dungeon, start, end)
        if not path_tiles:
            return 0
        entity_positions = self.find_all_tiles(dungeon, entity_type)
        count = 0
        for entity_pos in entity_positions:
            for path_pos in path_tiles:
                manhattan_dist = abs(entity_pos[0] - path_pos[0]) + abs(
                    entity_pos[1] - path_pos[1]
                )
                if manhattan_dist <= distance:
                    count += 1
                    break
        return count

    def get_path_tiles(
        self, dungeon: List[List[str]], start: Tuple[int, int], end: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        queue = deque([(start, [start])])
        visited = {start}
        while queue:
            (x, y), path = queue.popleft()
            if (x, y) == end:
                return path
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < self.height
                    and 0 <= ny < self.width
                    and (nx, ny) not in visited
                    and dungeon[nx][ny] != self.WALL
                ):
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))
        return []

    def selection(
        self, population: List[List[List[str]]], fitnesses: List[float]
    ) -> List[List[str]]:
        tournament_size = 5
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[
            tournament_fitnesses.index(max(tournament_fitnesses))
        ]
        return copy.deepcopy(population[winner_idx])

    def crossover(
        self, parent1: List[List[str]], parent2: List[List[str]]
    ) -> List[List[str]]:
        child = [[self.FLOOR for _ in range(self.width)] for _ in range(self.height)]
        crossover_row = random.randint(1, self.height - 2)
        for i in range(self.height):
            for j in range(self.width):
                if i < crossover_row:
                    child[i][j] = parent1[i][j]
                else:
                    child[i][j] = parent2[i][j]
        self.repair_dungeon(child)
        return child

    def mutate(self, dungeon: List[List[str]]) -> List[List[str]]:
        for i in range(self.height):
            for j in range(self.width):
                if random.random() < self.mutation_rate:
                    current_tile = dungeon[i][j]
                    if current_tile in [self.START, self.EXIT]:
                        continue
                    current_monsters = len(self.find_all_tiles(dungeon, self.MONSTER))
                    current_potions = len(self.find_all_tiles(dungeon, self.POTION))
                    current_treasures = len(self.find_all_tiles(dungeon, self.TREASURE))
                    mutation_type = random.random()
                    if mutation_type < 0.3:
                        if current_tile == self.WALL:
                            if random.random() < 0.3:
                                dungeon[i][j] = self.FLOOR
                        else:
                            dungeon[i][j] = self.WALL
                    elif (
                        mutation_type < 0.4
                        and current_monsters < self.target_monster_count + 2
                    ):
                        dungeon[i][j] = self.MONSTER
                    elif (
                        mutation_type < 0.5
                        and current_treasures < self.target_treasure_count + 1
                    ):
                        dungeon[i][j] = self.TREASURE
                    elif (
                        mutation_type < 0.6
                        and current_potions < self.target_potion_count + 1
                    ):
                        dungeon[i][j] = self.POTION
                    else:
                        if current_tile != self.WALL:
                            dungeon[i][j] = self.FLOOR
        self.repair_dungeon(dungeon)
        return dungeon

    def repair_dungeon(self, dungeon: List[List[str]]) -> None:
        starts = self.find_all_tiles(dungeon, self.START)
        exits = self.find_all_tiles(dungeon, self.EXIT)
        if len(starts) > 1:
            for i in range(1, len(starts)):
                dungeon[starts[i][0]][starts[i][1]] = self.FLOOR
        elif len(starts) == 0:
            x, y = 0, 0
            while dungeon[x][y] == self.WALL and x < self.height - 1:
                x += 1
            dungeon[x][y] = self.START
        if len(exits) > 1:
            for i in range(1, len(exits)):
                dungeon[exits[i][0]][exits[i][1]] = self.FLOOR
        elif len(exits) == 0:
            x, y = self.height - 1, self.width - 1
            while dungeon[x][y] == self.WALL and x > 0:
                x -= 1
            dungeon[x][y] = self.EXIT

        # Uses config vars here
        monsters = self.find_all_tiles(dungeon, self.MONSTER)
        if len(monsters) > self.target_monster_count + 2:
            monsters_to_remove = monsters[self.target_monster_count + 2 :]
            for pos in monsters_to_remove:
                dungeon[pos[0]][pos[1]] = self.FLOOR
        treasures = self.find_all_tiles(dungeon, self.TREASURE)
        if len(treasures) > self.target_treasure_count + 1:
            treasures_to_remove = treasures[self.target_treasure_count + 1 :]
            for pos in treasures_to_remove:
                dungeon[pos[0]][pos[1]] = self.FLOOR
        potions = self.find_all_tiles(dungeon, self.POTION)
        if len(potions) > self.target_potion_count + 1:
            potions_to_remove = potions[self.target_potion_count + 1 :]
            for pos in potions_to_remove:
                dungeon[pos[0]][pos[1]] = self.FLOOR

    def save_dungeon(self, dungeon: List[List[str]], stage_name: str) -> None:
        file_dir = path.dirname(__file__)
        stage_file = path.join(file_dir, "stages", f"{stage_name}.txt")
        with open(stage_file, "w") as f:
            for row in dungeon:
                f.write("".join(row) + "\n")
        print(f"Dungeon saved to {stage_file}")
        self.save_config(stage_name)

    def print_dungeon(self, dungeon: List[List[str]]) -> None:
        for row in dungeon:
            print("".join(row))
        print()

    def save_config(self, stage_name: str) -> None:
        import json

        file_dir = path.dirname(__file__)
        config_file = path.join(file_dir, "props", f"{stage_name}.json")
        with open(config_file, "w") as f:
            json.dump(self.game_config, f, indent=2)
        print(f"Config saved to {config_file}")

    # Configuration for the GAME (stats, points), separate from generator config
    game_config = {
        "PLAYER_MAX_HP": 30,
        "IS_PLAYER_HP_LIMIT": True,
        "ENEMY_POWER": 10,
        "ENEMY_POWER_MIN": 5,
        "ENEMY_POWER_MAX": 15,
        "IS_ENEMY_POWER_RANDOM": True,
        "POTION_POWER": 10,
        "DISTANCE_INF": 1000,
        "RENDER_WAIT_TIME": 0.05,
        "REWARDS": {
            "TURN": 1,
            "EXIT": 20,
            "KILL": 4,
            "TREASURE": 3,
            "POTION": 1,
            "DEAD": -20,
        },
    }
