from os import path

# from pathlib import Path
from typing import List, Optional, Tuple
import pygame


class StageRenderer:
    """Simple renderer for ASCII stage files using pygame.

    Responsibilities:
    - Load a stage from a text file (each char is a tile)
    - Provide a `render(surface, agent_pos)` method that draws the map
      and (optionally) the agent on top.
    """

    DEFAULT_COLORS = {
        "#": (80, 80, 80),
        ".": (200, 200, 200),
        "S": (200, 200, 200),
        "E": (100, 200, 100),
        "T": (230, 190, 20),
        "M": (200, 50, 50),
        "P": (150, 50, 180),
    }

    # mapping from map character to asset filename (inside pcg/assets)
    DEFAULT_SPRITES = {
        "#": "wall.png",
        ".": "empty.png",
        "S": "entrance.png",
        "E": "exit.png",
        "T": "chest.png",
        "M": "monster.png",
        "P": "potion.png",
    }

    def __init__(self, stage_name: str, window_size: int = 512):
        self.window_size = window_size

        self.grid: List[List[str]] = []
        self.width = 0
        self.height = 0
        self.tile_size = 16
        self.start_pos: Optional[Tuple[int, int]] = None
        self.sprites: dict[str, Optional[pygame.Surface]] = {}

        current_dir = path.dirname(__file__)
        self._load_file(current_dir, stage_name)
        self._load_sprites(current_dir)

    def _load_file(self, current_dir: str, stage_name: str):
        stage_file = path.join(current_dir, "stages", f"{stage_name}.txt")
        with open(stage_file, "r") as f:
            texts = [s.strip() for s in f]
        self._load_from_lines(texts)

    def _load_from_lines(self, lines: List[str]):
        self.grid = [list(line) for line in lines]
        self.height = len(self.grid)
        self.width = max((len(r) for r in self.grid), default=0)
        # compute tile size so map fits window
        if self.width and self.height:
            self.tile_size = max(
                4, min(self.window_size // max(self.width, self.height), 64)
            )
        # find start
        self.start_pos = None
        for y, row in enumerate(self.grid):
            for x, ch in enumerate(row):
                if ch == "S":
                    self.start_pos = (x, y)
                    return

    def _load_sprites(self, current_dir: str):
        """Load sprite images from the given assets directory.

        Missing files are ignored (fall back to colored tiles).
        """
        for ch, fname in self.DEFAULT_SPRITES.items():
            asset_path = path.join(current_dir, "assets", fname)
            try:
                surf = pygame.image.load(asset_path)
                # only call convert_alpha/convert if a video surface is available
                if pygame.display.get_surface() is not None:
                    try:
                        surf = surf.convert_alpha()
                    except Exception:
                        surf = surf.convert()
                self.sprites[ch] = surf
            except Exception:
                self.sprites[ch] = None

        # load agent sprite (hero) if available
        agent_candidates = ["hero.png", "deadhero.png"]
        for name in agent_candidates:
            p = path.join(current_dir, "assets", name)
            try:
                surf = pygame.image.load(p)
                if pygame.display.get_surface() is not None:
                    try:
                        surf = surf.convert_alpha()
                    except Exception:
                        surf = surf.convert()
                self.sprites["_agent"] = surf
                break
            except Exception:
                continue
        if "_agent" not in self.sprites:
            self.sprites["_agent"] = None

    def render(
        self,
        surface: pygame.Surface,
        agent_pos: Optional[Tuple[int, int]] = None,
        agent_hp: Optional[int] = None,
        agent_max_hp: Optional[int] = None,
    ):
        """Draw the stage into the provided surface. Coordinates are grid-based (x,y).

        Agent is drawn as a filled circle on top of tiles.
        """
        if not self.grid:
            return

        # draw floor (empty) under every tile first
        floor_sprite = self.sprites.get(".")
        floor_color = self.DEFAULT_COLORS.get(".", (200, 200, 200))

        for y, row in enumerate(self.grid):
            for x in range(self.width):
                ch = row[x] if x < len(row) else " "
                rect = pygame.Rect(
                    x * self.tile_size,
                    y * self.tile_size,
                    self.tile_size,
                    self.tile_size,
                )

                # draw floor beneath everything
                if floor_sprite:
                    try:
                        floor_img = pygame.transform.smoothscale(
                            floor_sprite, (self.tile_size, self.tile_size)
                        )
                    except Exception:
                        floor_img = pygame.transform.scale(
                            floor_sprite, (self.tile_size, self.tile_size)
                        )
                    surface.blit(floor_img, rect.topleft)
                else:
                    pygame.draw.rect(surface, floor_color, rect)

                # draw the tile sprite or colored tile on top
                sprite = self.sprites.get(ch)
                if sprite and ch != ".":
                    try:
                        img = pygame.transform.smoothscale(
                            sprite, (self.tile_size, self.tile_size)
                        )
                    except Exception:
                        img = pygame.transform.scale(
                            sprite, (self.tile_size, self.tile_size)
                        )
                    surface.blit(img, rect.topleft)
                else:
                    # if no sprite (or ch is '.' which is already drawn as floor), draw overlay color for non-floor
                    if ch != "." and ch != " ":
                        color = self.DEFAULT_COLORS.get(ch, (50, 50, 50))
                        pygame.draw.rect(surface, color, rect)

                # draw grid lines for clarity
                pygame.draw.rect(surface, (220, 220, 220), rect, 1)

        # draw agent on top
        if agent_pos is not None:
            ax, ay = agent_pos
            sprite = self.sprites.get("_agent")
            if sprite:
                try:
                    img = pygame.transform.smoothscale(
                        sprite, (self.tile_size, self.tile_size)
                    )
                except Exception:
                    img = pygame.transform.scale(
                        sprite, (self.tile_size, self.tile_size)
                    )
                surface.blit(img, (ax * self.tile_size, ay * self.tile_size))
            else:
                cx = int((ax + 0.5) * self.tile_size)
                cy = int((ay + 0.5) * self.tile_size)
                radius = max(2, self.tile_size // 3)
                pygame.draw.circle(surface, (50, 120, 200), (cx, cy), radius)

            # draw HP bar above agent if hp info is provided
            try:
                if agent_hp is not None and agent_max_hp is not None and agent_max_hp > 0:
                    # compute bar dimensions
                    pad = max(2, self.tile_size // 10)
                    bar_w = max(8, int(self.tile_size * 0.8))
                    bar_h = max(4, self.tile_size // 8)
                    bar_x = ax * self.tile_size + (self.tile_size - bar_w) // 2
                    bar_y = ay * self.tile_size - bar_h - pad
                    # if not enough space above, draw inside the tile at top
                    if bar_y < 0:
                        bar_y = ay * self.tile_size + pad

                    # background
                    bg_rect = pygame.Rect(bar_x, bar_y, bar_w, bar_h)
                    pygame.draw.rect(surface, (60, 60, 60), bg_rect)

                    # health fraction
                    frac = max(0.0, min(1.0, float(agent_hp) / float(agent_max_hp)))
                    fill_w = int(frac * (bar_w - 2))
                    fill_rect = pygame.Rect(bar_x + 1, bar_y + 1, fill_w, bar_h - 2)
                    # color gradient: red -> yellow -> green
                    if frac > 0.66:
                        color = (50, 200, 50)
                    elif frac > 0.33:
                        color = (230, 200, 50)
                    else:
                        color = (200, 60, 60)
                    if fill_w > 0:
                        pygame.draw.rect(surface, color, fill_rect)

                    # border
                    pygame.draw.rect(surface, (30, 30, 30), bg_rect, 1)
            except Exception:
                pass
