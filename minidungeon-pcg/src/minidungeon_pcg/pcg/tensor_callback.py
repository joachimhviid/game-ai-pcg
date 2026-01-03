import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class CustomTensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.tile_colors = {
            0: (200, 200, 200),  # FLOOR = "."
            1: (0, 0, 0),  # WALL = "#"
            2: (0, 255, 0),  # START = "S"
            3: (255, 0, 0),  # EXIT = "E"
            4: (0, 0, 255),  # MONSTER = "M"
            5: (222, 75, 207),  # POTION = "P"
            6: (250, 233, 42),  # TREASURE = "T"
        }

    def _on_step(self) -> bool:
        # 1. Log a simple scalar (e.g., a random value or a calculation)
        # value = np.random.random()
        # self.logger.record("custom/random_value", value)

        # 2. Access environment info (if your env returns it in the 'info' dict)
        # This is useful for logging internal environment states
        if len(self.locals["infos"]) > 0:
            # Get the info dict from the first environment
            info = self.locals["infos"][0]
            if "delta_reward" in info:
                self.logger.record("env/delta_reward", info["delta_reward"])
            if "target_reward" in info:
                self.logger.record("env/target_reward", info["target_reward"])
            if "sim_reward" in info:
                self.logger.record("env/sim_reward", info["sim_reward"])
            if "distance_to_exit" in info:
                self.logger.record("env/distance_to_exit", info["distance_to_exit"])
            if "dungeon" in info:
                fig = self._get_dungeon_plot()
                self.logger.record("env/current_level", Figure(fig, close=True))
                plt.close(fig)

        return True

    def _get_dungeon_plot(self):
        current_dungeon = self.locals["infos"][0]["dungeon"]
        # current_dungeon = self.training_env.get_attr("dungeon")[0]
        map_data = current_dungeon.squeeze()
        rows, cols = map_data.shape
        image = np.zeros((rows, cols, 3), dtype=np.uint8)

        for tile_id, color in self.tile_colors.items():
            mask = map_data == tile_id
            image[mask] = color

        fig, ax = plt.subplots(figsize=(9, 10))
        ax.imshow(image, interpolation="nearest")
        ax.axis("off")
        ax.set_title(f"Observed Level {self.num_timesteps}")

        return fig
