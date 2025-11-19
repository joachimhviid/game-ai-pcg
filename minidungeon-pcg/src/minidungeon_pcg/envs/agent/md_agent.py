import numpy as np

class MdAgent:
    def __init__(self) -> None:
        """Agent helper that implements environment actions and movement logic.

        This class is intentionally simple: it exposes `step(action, agent_pos, grid, w, h)`
        which returns (new_pos, reward, terminated, truncated, info, grid).
        """

    def step(self, action: int, agent_pos, grid, w, h):
        """Execute an action and update the grid/state.

        Parameters:
        - action: discrete action index
        - agent_pos: (x,y) tuple current agent position
        - grid: list[list[str]] mutable grid; modified in-place for pickups
        - w,h: grid width and height

        Returns:
        - new_pos (x,y)
        - reward (float)
        - terminated (bool)
        - truncated (bool)
        - info (dict)
        - grid (possibly modified)
        """
        
        act_idx = int(action)

        reward = -0.01
        terminated = False
        truncated = False

        if agent_pos is None:
            agent_pos = (0, 0)

        ax, ay = agent_pos

        # movement deltas
        deltas = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

        if act_idx in deltas:
            dx, dy = deltas[act_idx]
            nx, ny = ax + dx, ay + dy
            # check bounds
            if 0 <= nx < w and 0 <= ny < h:
                target = grid[ny][nx] if nx < len(grid[ny]) else " "
                if target != "#":
                    if target == "M":
                        reward += -5.0
                        terminated = True
                        agent_pos = (nx, ny)
                        grid[ny][nx] = "."
                    else:
                        agent_pos = (nx, ny)
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

        info = {"action": act_idx}
        return agent_pos, float(reward), bool(terminated), bool(truncated), info, grid
