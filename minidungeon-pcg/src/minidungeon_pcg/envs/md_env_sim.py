from typing import Any
import gymnasium as gym
from minidungeon_pcg.envs.agent.md_treasure_agent import MdTreasureAgent
import numpy as np


class MdEnvSim(gym.Env[np.ndarray, np.ndarray]):
    def __init__(self, level: np.ndarray, debug: bool = False):
        self.debug = debug
        self.level = level
        self.level_width, self.level_height = level.shape

        self.agent = MdTreasureAgent(debug=self.debug)
        self._closed = False

        # keep an editable copy of the map so env dynamics (treasure pickup etc.)
        # can modify it independent of the original stage file. StageRenderer
        # will render whatever is in `self.stage_renderer.grid`, so we keep
        # a deep copy to restore on reset.
        self._initial_grid = self.level.copy()
        # self._initial_grid = [list(r) for r in self.level]

        # actions: gym-md style - a length-7 float vector where the env picks
        # the highest-scoring high-level action (head-to-monster, head-to-treasure, ...)
        # We'll accept either a length-7 float vector or an integer discrete action.
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,))
        self.observation_space = gym.spaces.Box(
            low=0, high=1000, shape=(8,), dtype=np.int32
        )

    def step(self, action):
        if self._closed:
            raise RuntimeError("Environment is closed")

        # delegate action handling to the agent
        grid = self.level
        w = self.level_width
        h = self.level_height

        selected = self.agent.select_action(action, grid)
        new_pos, reward, terminated, truncated, info, new_grid, new_hp = (
            self.agent.take_action(selected, grid, w, h)
        )

        # adopt agent results
        self.level = new_grid

        obs = self._get_observation()
        info = {"agent_pos": new_pos, "agent_hp": new_hp, **info}
        return obs, float(reward), bool(terminated), bool(truncated), info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        # restore a fresh copy of the initial grid
        self.level = self._initial_grid.copy()
        rows, cols = np.where(self.level == "S")

        # No starting location found
        if len(rows) == 0:
            self.agent.position = (0, 0)
        else:
            self.agent.position = (rows[0], cols[0])

        # reset HP
        self.agent.hp = self.agent.max_hp

        obs = self._get_observation()
        info = {"agent_pos": self.agent.position}
        return obs, info

    def _get_observation(self):
        # Build an 8-element observation vector:
        # 0: distance to nearest monster
        # 1: distance to nearest treasure
        # 2: distance to treasure using paths that avoid monsters
        # 3: distance to nearest potion
        # 4: distance to potion (avoid monsters)
        # 5: distance to exit
        # 6: distance to exit (avoid monsters)
        # 7: agent HP

        grid = list(self.level)

        # use the agent-attached Pather helper for distance queries
        pather = self.agent.pather

        # determine agent start position
        start = self.agent.position if self.agent.position is not None else (0, 0)

        if pather is None or not grid:
            # fallback: unreachable distances
            d_mon = d_tre = d_tre_avoid = d_pot = d_pot_avoid = d_exit = (
                d_exit_avoid
            ) = 1000
        else:
            d_mon = pather.distance_to_nearest(grid, start, {"M"}, avoid_monsters=False)
            d_tre = pather.distance_to_nearest(grid, start, {"T"}, avoid_monsters=False)
            d_tre_avoid = pather.distance_to_nearest(
                grid, start, {"T"}, avoid_monsters=True
            )
            d_pot = pather.distance_to_nearest(grid, start, {"P"}, avoid_monsters=False)
            d_pot_avoid = pather.distance_to_nearest(
                grid, start, {"P"}, avoid_monsters=True
            )
            d_exit = pather.distance_to_nearest(
                grid, start, {"E"}, avoid_monsters=False
            )
            d_exit_avoid = pather.distance_to_nearest(
                grid, start, {"E"}, avoid_monsters=True
            )

        obs = np.array(
            [
                d_mon,
                d_tre,
                d_tre_avoid,
                d_pot,
                d_pot_avoid,
                d_exit,
                d_exit_avoid,
                self.agent.hp,
            ],
            dtype=self.observation_space.dtype,
        )

        return obs

    def _get_info(self):
        return {}

    def close(self):
        try:
            return super().close()
        except Exception:
            return None
