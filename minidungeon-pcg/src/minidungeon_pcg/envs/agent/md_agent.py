import numpy as np
import random
from .pather import Pather


class MdAgent:
    def __init__(self, max_hp: int, debug: bool = False) -> None:
        """Agent helper that implements environment actions and movement logic.

        A `Pather` instance is attached as `self.pather` to provide BFS-based
        pathfinding helpers for higher-level action decisions.
        """
        self.max_hp = max_hp
        self.hp = self.max_hp
        self.position = None
        self.pather = Pather()
        self.debug = debug

    def next_action_to(self, grid, target_chars: set, avoid_monsters: bool = False) -> int:
        """Return a discrete action index that moves one step toward the nearest
        tile matching `target_chars`. Uses the attached `Pather`.
        """
        if self.position is None:
            return 0
        return self.pather.next_action(grid, self.position, set(target_chars), avoid_monsters=avoid_monsters)

    def path_to(self, grid, target_chars: set, avoid_monsters: bool = False):
        """Return full shortest path (list of (x,y)) to nearest `target_chars`.
        """
        if self.position is None:
            return []
        return self.pather.shortest_path(grid, self.position, set(target_chars), avoid_monsters=avoid_monsters)

    def select_action(self, action_vector, grid):
        """Select a high-level action index from a length-7 float vector.

        The method ranks the vector values (ties randomized) and returns the
        first high-level index for which a feasible low-level move exists.

        Returns the selected high-level index (0..6) or `None` if no feasible
        action is found.
        """
        arr = np.asarray(action_vector, dtype=float)
        if arr.ndim != 1 or arr.size != 7:
            raise ValueError("action must be a length-7 float vector")

        vals = arr.tolist()
        groups = {}
        for i, v in enumerate(vals):
            groups.setdefault(v, []).append(i)

        unique_vals = sorted(groups.keys(), reverse=True)
        candidate_indices = []
        for v in unique_vals:
            inds = groups[v].copy()
            random.shuffle(inds)
            candidate_indices.extend(inds)

        mapping = {
            0: ({"M"}, False),
            1: ({"T"}, False),
            2: ({"T"}, True),
            3: ({"P"}, False),
            4: ({"P"}, True),
            5: ({"E"}, False),
            6: ({"E"}, True),
        }

        start = self.position if self.position is not None else (0, 0)
        for idx in candidate_indices:
            target_chars, avoid = mapping[idx]
            move = self.pather.next_action(grid, start, set(target_chars), avoid_monsters=avoid)
            if move != 0:
                return idx

        return None

    def take_action(self, selected_action, grid, w, h):
        """Execute the given high-level `selected_action`.

        `selected_action` should be an int in 0..6 (matching the mapping used
        by `select_action`). If `selected_action` is None or no feasible low-
        level move exists, the agent performs a noop.

        Returns the same tuple as `step` previously did: (position, reward,
        terminated, truncated, info, grid, hp).
        """
        # resolve high-level into low-level move
        mapping = {
            0: ({"M"}, False),
            1: ({"T"}, False),
            2: ({"T"}, True),
            3: ({"P"}, False),
            4: ({"P"}, True),
            5: ({"E"}, False),
            6: ({"E"}, True),
        }

        if selected_action is None or selected_action not in mapping:
            resolved_low_level = 0
        else:
            target_chars, avoid = mapping[selected_action]
            start = self.position if self.position is not None else (0, 0)
            resolved_low_level = self.pather.next_action(grid, start, set(target_chars), avoid_monsters=avoid)

        act_idx = int(resolved_low_level)

        # now perform the low-level action (same logic as before)
        reward = -0.01
        terminated = False
        truncated = False

        if self.position is None:
            self.position = (0, 0)

        ax, ay = self.position
        deltas = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

        if act_idx in deltas:
            dx, dy = deltas[act_idx]
            nx, ny = ax + dx, ay + dy
            if 0 <= nx < w and 0 <= ny < h:
                target = grid[ny][nx] if nx < len(grid[ny]) else " "
                if target != "#":
                    if target == "M":
                        self.hp -= 5
                        reward += -5.0
                        self.position = (nx, ny)
                        grid[ny][nx] = "."
                        if self.hp <= 0:
                            terminated = True
                    else:
                        self.position = (nx, ny)
                        if target == "T":
                            reward += 1.0
                            grid[ny][nx] = "."
                        if target == "E":
                            reward += 10.0
                            terminated = True
                else:
                    reward += -0.1
            else:
                reward += -0.1
        elif act_idx == 5:
            cx, cy = ax, ay
            if 0 <= cx < w and 0 <= cy < h:
                if grid[cy][cx] == "T":
                    reward += 1.0
                    grid[cy][cx] = "."

        # clamp reward to reasonable bounds and optionally log for debugging
        reward = float(reward)
        reward = max(-100.0, min(100.0, reward))
        if getattr(self, "debug", False):
            try:
                print(
                    "MdAgent.take_action: selected=", selected_action,
                    "low_level=", act_idx,
                    "reward=", reward,
                    "pos=", self.position,
                    "hp=", self.hp,
                )
            except Exception:
                pass

        info = {"selected_high_level": selected_action, "action": act_idx}
        return self.position, reward, bool(terminated), bool(truncated), info, grid, self.hp

