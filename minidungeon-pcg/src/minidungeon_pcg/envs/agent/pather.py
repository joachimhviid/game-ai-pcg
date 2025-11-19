from collections import deque
from typing import Iterable, List, Optional, Sequence, Set, Tuple


Position = Tuple[int, int]


class Pather:
    """Pathfinding helper for grid-based stages.

    Grid format expected: list of rows, each row is a sequence of characters (str).
    Coordinates are (x, y) with x indexing columns and y indexing rows.

    Example usage:
        p = Pather()
        path = p.shortest_path(grid, (sx,sy), {"E"})
        next_act = p.next_action(grid, (sx,sy), (tx,ty))

    Action mapping returned by `next_action` follows the agent's discrete
    convention used in this project: 1 up, 2 down, 3 left, 4 right. If no
    movement is possible or no path exists the method returns 0 (noop).
    """

    def __init__(self) -> None:
        pass

    def _grid_size(self, grid: Sequence[Sequence[str]]) -> Tuple[int, int]:
        h = len(grid)
        w = 0 if h == 0 else max(len(row) for row in grid)
        return w, h

    def bfs(self, grid: Sequence[Sequence[str]], start: Position, avoid_monsters: bool = False):
        """Run BFS from `start` and return (distances, predecessors).

        - distances: dict[(x,y)] -> distance
        - prev: dict[(x,y)] -> previous (x,y) on path from start
        """
        w, h = self._grid_size(grid)
        sx, sy = start
        if w == 0 or h == 0:
            return {}, {}

        distances = {}
        prev = {}
        q = deque()
        q.append((sx, sy))
        distances[(sx, sy)] = 0

        while q:
            x, y = q.popleft()
            d = distances[(x, y)]
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if (nx, ny) in distances:
                    continue
                tch = grid[ny][nx] if nx < len(grid[ny]) else " "
                if tch == "#":
                    continue
                if avoid_monsters and tch == "M":
                    continue
                distances[(nx, ny)] = d + 1
                prev[(nx, ny)] = (x, y)
                q.append((nx, ny))

        return distances, prev

    def shortest_path(self, grid: Sequence[Sequence[str]], start: Position,
                      target_chars: Set[str], avoid_monsters: bool = False) -> List[Position]:
        """Return shortest path (including start and goal) to the nearest tile
        whose character is in `target_chars`. If none found, returns empty list.
        """
        distances, prev = self.bfs(grid, start, avoid_monsters=avoid_monsters)
        if not distances:
            return []

        # find nearest target cell
        target_pos: Optional[Position] = None
        best_d = None
        h = len(grid)
        for y in range(h):
            row = grid[y]
            for x in range(len(row)):
                if (x, y) == start:
                    continue
                if row[x] in target_chars and (x, y) in distances:
                    d = distances[(x, y)]
                    if best_d is None or d < best_d:
                        best_d = d
                        target_pos = (x, y)

        if target_pos is None:
            return []

        # reconstruct path
        path: List[Position] = []
        cur = target_pos
        while True:
            path.append(cur)
            if cur == start:
                break
            cur = prev.get(cur)
            if cur is None:
                # shouldn't happen if distances contains target
                return []

        path.reverse()
        return path

    def next_step(self, grid: Sequence[Sequence[str]], start: Position,
                  target_chars: Set[str], avoid_monsters: bool = False) -> Optional[Position]:
        """Return the immediate next position (x,y) along a shortest path to
        the nearest tile matching `target_chars`. Returns None if no path.
        """
        path = self.shortest_path(grid, start, target_chars, avoid_monsters=avoid_monsters)
        if len(path) < 2:
            return None
        return path[1]

    def next_action(self, grid: Sequence[Sequence[str]], start: Position,
                    target_chars: Set[str], avoid_monsters: bool = False) -> int:
        """Return the discrete action index to move one step toward the nearest
        `target_chars` tile. Action mapping: 1 up, 2 down, 3 left, 4 right.
        Returns 0 (noop) if no move is possible or no path exists.
        """
        nxt = self.next_step(grid, start, target_chars, avoid_monsters=avoid_monsters)
        if nxt is None:
            return 0
        sx, sy = start
        nx, ny = nxt
        dx, dy = nx - sx, ny - sy
        if (dx, dy) == (0, -1):
            return 1
        if (dx, dy) == (0, 1):
            return 2
        if (dx, dy) == (-1, 0):
            return 3
        if (dx, dy) == (1, 0):
            return 4
        return 0

    def distance_to_nearest(self, grid: Sequence[Sequence[str]], start: Position,
                            target_chars: Set[str], avoid_monsters: bool = False) -> int:
        """Return integer distance to nearest target (1000 if unreachable).
        """
        distances, _ = self.bfs(grid, start, avoid_monsters=avoid_monsters)
        if not distances:
            return 1000
        best = None
        h = len(grid)
        for y in range(h):
            row = grid[y]
            for x in range(len(row)):
                if row[x] in target_chars and (x, y) in distances and not (x, y) == start:
                    d = distances[(x, y)]
                    if best is None or d < best:
                        best = d
        return 1000 if best is None else best
