from typing import Any
from pathlib import Path
import gymnasium as gym
from minidungeon_pcg.envs.agent.md_agent import MdAgent
from minidungeon_pcg.pcg.stage_renderer import StageRenderer
import numpy as np
import pygame


class MdEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, stage_name: str, render_mode=None):
        self.render_mode = render_mode

        self.window_size = 512
        self.window = None
        self.clock = None

        self.agent = MdAgent()
        self.agent_pos = None
        self._closed = False

        # ensure a Path is passed to the renderer
        self.stage_renderer = StageRenderer(stage_name, window_size=self.window_size)
        # keep an editable copy of the map so env dynamics (treasure pickup etc.)
        # can modify it independent of the original stage file. StageRenderer
        # will render whatever is in `self.stage_renderer.grid`, so we keep
        # a deep copy to restore on reset.
        self._initial_grid = [list(r) for r in self.stage_renderer.grid]
        # actions are discrete: 0 noop, 1 up, 2 down, 3 left, 4 right, 5 pickup, 6 noop
        self.action_space = gym.spaces.Discrete(7)
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

        new_pos, reward, terminated, truncated, info, new_grid = self.agent.step(
            action, self.agent_pos, grid, w, h
        )

        # adopt agent results
        self.agent_pos = new_pos
        self.stage_renderer.grid = new_grid

        obs = self._get_observation()
        # include agent_pos in info for convenience
        info = {"agent_pos": self.agent_pos, **info}
        return obs, float(reward), bool(terminated), bool(truncated), info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        # restore a fresh copy of the initial grid
        self.stage_renderer.grid = [list(r) for r in self._initial_grid]

        if hasattr(self, "stage_renderer") and self.stage_renderer.start_pos is not None:
            self.agent_pos = self.stage_renderer.start_pos
        else:
            self.agent_pos = (0, 0)

        obs = self._get_observation()
        info = {"agent_pos": self.agent_pos}
        return obs, info

    def _get_observation(self):
        # placeholder observation -- adapt to your actual observation format
        shape = (
            tuple(self.observation_space.shape)
            if self.observation_space.shape is not None
            else (0,)
        )
        return np.zeros(shape, dtype=self.observation_space.dtype)

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
            self.stage_renderer.render(canvas, agent_pos=self.agent_pos)
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
