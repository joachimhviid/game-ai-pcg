from typing import Any
from pathlib import Path
import gymnasium as gym
from minidungeon_pcg.envs.agent.md_agent import MdAgent
from minidungeon_pcg.pcg.stage_renderer import StageRenderer
import numpy as np
import pygame
import random


class MdEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, stage_name: str, render_mode=None, debug: bool = False):
        self.render_mode = render_mode
        self.debug = debug

        self.window_size = 512
        self.window = None
        self.clock = None

        self.agent = MdAgent(10, debug=self.debug)
        self._closed = False

        self.stage_renderer = StageRenderer(stage_name, window_size=self.window_size)
        
        # keep an editable copy of the map so env dynamics (treasure pickup etc.)
        # can modify it independent of the original stage file. StageRenderer
        # will render whatever is in `self.stage_renderer.grid`, so we keep
        # a deep copy to restore on reset.
        self._initial_grid = [list(r) for r in self.stage_renderer.grid]
        
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
        grid = self.stage_renderer.grid
        w = self.stage_renderer.width
        h = self.stage_renderer.height
        
        selected = self.agent.select_action(action, grid)
        new_pos, reward, terminated, truncated, info, new_grid, new_hp = self.agent.take_action(selected, grid, w, h)

        # adopt agent results
        self.stage_renderer.grid = new_grid

        obs = self._get_observation()
        info = {"agent_pos": new_pos, "agent_hp": new_hp, **info}
        return obs, float(reward), bool(terminated), bool(truncated), info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        # restore a fresh copy of the initial grid
        self.stage_renderer.grid = [list(r) for r in self._initial_grid]

        if self.stage_renderer.start_pos is not None:
            self.agent.position = self.stage_renderer.start_pos
        else:
            self.agent.position = (0, 0)

        # reset HP
        self.agent.hp = self.agent.max_hp

        obs = self._get_observation()
        info = {"agent_pos": self.agent.position}
        return obs, info

    # action resolution is handled by the agent (MdAgent.step)

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

        grid = self.stage_renderer.grid

        # use the agent-attached Pather helper for distance queries
        pather = self.agent.pather

        # determine agent start position
        start = self.agent.position if self.agent.position is not None else (0, 0)

        if pather is None or not grid:
            # fallback: unreachable distances
            d_mon = d_tre = d_tre_avoid = d_pot = d_pot_avoid = d_exit = d_exit_avoid = 1000
        else:
            d_mon = pather.distance_to_nearest(grid, start, {"M"}, avoid_monsters=False)
            d_tre = pather.distance_to_nearest(grid, start, {"T"}, avoid_monsters=False)
            d_tre_avoid = pather.distance_to_nearest(grid, start, {"T"}, avoid_monsters=True)
            d_pot = pather.distance_to_nearest(grid, start, {"P"}, avoid_monsters=False)
            d_pot_avoid = pather.distance_to_nearest(grid, start, {"P"}, avoid_monsters=True)
            d_exit = pather.distance_to_nearest(grid, start, {"E"}, avoid_monsters=False)
            d_exit_avoid = pather.distance_to_nearest(grid, start, {"E"}, avoid_monsters=True)

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

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        if self._closed:
            return None

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # let the stage renderer draw the map and agent
        try:
            self.stage_renderer.render(canvas, agent_pos=self.agent.position)
        except Exception:
            # fail silently for rendering so render() doesn't crash the caller
            pass

        # blit to window and update display
        if self.window is not None:
            self.window.blit(canvas, canvas.get_rect())

            # handle events so window remains responsive
            for event in pygame.event.get():
                try:
                    if event.type == pygame.QUIT:
                        # close the env and stop rendering
                        self.close()
                        return None
                    # allow Escape to close window too
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        self.close()
                        return None
                except Exception:
                    # be resilient to unexpected event attributes
                    continue

            pygame.display.update()
        if self.clock is not None:
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        # tear down pygame window and subsystems safely
        try:
            if self.window is not None:
                try:
                    pygame.display.quit()
                except Exception:
                    pass
            try:
                pygame.quit()
            except Exception:
                pass
        finally:
            self.window = None
            self.clock = None
            self._closed = True

        try:
            return super().close()
        except Exception:
            return None
