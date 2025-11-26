from os import path
from typing import List
from gym_md.envs.grid import Grid
from minidungeon_pcg.pcg.setting import PcgSetting


class PcgGrid(Grid):
    def __init__(self, stage_name: str, setting: PcgSetting) -> None:
        self.texts = PcgGrid.read_grid_as_list_from_stage_name(  # type: ignore
            stage_name
        )
        self.setting = setting  # type: ignore
        self.H = len(self.texts)  # type: ignore
        self.W = len(self.texts[0])  # type: ignore
        self.g = [[0] * self.W for _ in range(self.H)]  # type: ignore

        self.reset()

    @staticmethod
    def read_grid_as_list_from_stage_name(stage_name: str) -> List[str]:
        file_dir = path.dirname(__file__)
        stage_file = path.join(file_dir, "stages", f"{stage_name}.txt")
        with open(stage_file, "r") as f:
            texts = [s.strip() for s in f]
        return texts
