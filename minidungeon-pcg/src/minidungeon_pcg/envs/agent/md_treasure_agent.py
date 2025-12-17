import numpy as np
from minidungeon_pcg.envs.agent.md_agent import MdAgent
from minidungeon_pcg.envs.settings import Settings


class MdTreasureAgent(MdAgent):
    def __init__(self, debug: bool = False) -> None:
        super().__init__(debug)
        self.standard_vector = np.array(
            # Action 3 (Go to Potion) is given a slight edge over Action 0 (Go to Monster)
            # to break ties logically instead of randomly.
            [0.0, 0.9, 1.0, 0.1, 0.7, 0.6, 0.8],
            dtype=np.float32,
        )
        self.survival_vector = np.array(
            # In survival mode, the preference for a potion should be even clearer.
            [0.0, 0.8, 0.9, 0.2, 1.0, 0.5, 0.7],
            dtype=np.float32,
        )
        self.is_survival_mode = False

    def select_action(self, action_vector, grid):
        if self.is_survival_mode:
            if self.can_survive_fight():
                self.is_survival_mode = False
            else:
                return super().select_action(self.survival_vector, grid)

        intended_action = super().select_action(self.standard_vector, grid)

        if intended_action is not None and self.position is not None:
            target, avoid = self.action_mapping[intended_action]
            move = self.pather.next_action(
                grid, self.position, target, avoid_monsters=avoid
            )

            if move in self.deltas:
                x, y = self.position
                dx, dy = self.deltas[move]
                next_x, next_y = x + dx, y + dy

                target_tile = grid[next_y][next_x]
                if target_tile == "M" and not self.can_survive_fight():
                    surival_move = self.pather.next_action(
                        grid, self.position, {"P"}, avoid_monsters=True
                    )
                    if surival_move != 0:
                        self.is_survival_mode = True
                        return super().select_action(self.survival_vector, grid)
                    else:
                        return intended_action

        return intended_action

    def can_survive_fight(self) -> bool:
        return self.hp > Settings.MONSTER_DAMAGE
