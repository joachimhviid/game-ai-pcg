from copy import deepcopy
from typing import Dict, Final, List
from gym_md.envs import definition
from gym_md.envs.setting import Setting
from gym_md.envs.config.props_config import PropsConfig, RewardsConfig
from os import path, listdir
from platform import system as platform_system
import json


class PcgSetting(Setting):
    def __init__(self, stage_name: str):
        self.STAGE_NAME = stage_name  # type: ignore
        self.GRID_CHARACTERS = definition.GRID_CHARACTERS  # type: ignore
        self.OBSERVATIONS = definition.OBSERVATIONS  # type: ignore
        self.ACTIONS = definition.ACTIONS  # type: ignore
        self.CHARACTER_TO_NUM = Setting.list_to_dict(  # type: ignore
            self.GRID_CHARACTERS
        )
        self.NUM_TO_CHARACTER = Setting.swap_dict(self.CHARACTER_TO_NUM)  # type: ignore
        self.OBSERVATION_TO_NUM = Setting.list_to_dict(  # type: ignore
            self.OBSERVATIONS
        )
        self.NUM_TO_OBSERVATION = Setting.swap_dict(  # type: ignore
            self.OBSERVATION_TO_NUM
        )
        self.ACTION_TO_NUM = Setting.list_to_dict(self.ACTIONS)  # type: ignore
        self.NUM_TO_ACTION = Setting.swap_dict(self.ACTION_TO_NUM)  # type: ignore

        props_config = PcgSetting.read_settings(stage_name)
        self.PLAYER_MAX_HP = props_config.PLAYER_MAX_HP  # type: ignore
        self.IS_PLAYER_HP_LIMIT = props_config.IS_PLAYER_HP_LIMIT
        self.ENEMY_POWER = props_config.ENEMY_POWER
        self.ENEMY_POWER_MIN = props_config.ENEMY_POWER_MIN
        self.ENEMY_POWER_MAX = props_config.ENEMY_POWER_MAX
        self.IS_ENEMY_POWER_RANDOM = props_config.IS_ENEMY_POWER_RANDOM
        self.POTION_POWER = props_config.POTION_POWER
        self.DISTANCE_INF = props_config.DISTANCE_INF
        self.RENDER_WAIT_TIME = props_config.RENDER_WAIT_TIME
        self.REWARDS: RewardsConfig = deepcopy(props_config.REWARDS)
        self.ORIGINAL_REWARDS: RewardsConfig = deepcopy(props_config.REWARDS)

    @staticmethod
    def read_settings(stage_name: str) -> PropsConfig:
        file_dir: str = path.dirname(__file__)
        target_stage_file: str = f"{stage_name}.json"
        json_path: str = path.join(file_dir, "props", target_stage_file)

        if platform_system().lower() == "linux":
            prop_files_dir: list = listdir(path.join(file_dir, "props"))
            prop_files_dir_lowercased: list = [
                file_name.lower() for file_name in prop_files_dir
            ]
            actual_stage_file: str = prop_files_dir[
                prop_files_dir_lowercased.index(target_stage_file.lower())
            ]
            # do not re-annotate json_path to avoid mypy no-redef warning
            json_path = path.join(file_dir, "props", actual_stage_file)

        with open(json_path, "r") as f:
            data = json.load(f)
            props_config = PropsConfig(**data)
        return props_config
