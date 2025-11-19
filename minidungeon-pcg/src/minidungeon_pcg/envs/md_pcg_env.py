from collections import defaultdict
from random import Random
from typing import DefaultDict, Final, Optional
import gymnasium as gym
import numpy as np
from gym_md.envs.md_env import MdEnvBase
from gym_md.envs.agent.agent import Agent
from gym_md.envs.renderer.renderer import Renderer
from minidungeon_pcg.pcg.grid import PcgGrid
from minidungeon_pcg.pcg.setting import PcgSetting


class MdPcgEnv(MdEnvBase):
    metadata = {"render_modes": ["human"]}

    def __init__(self, stage_name: str, render_mode: Optional[str]):
        self.render_mode = render_mode
        self.random = Random()
        self.stage_name = stage_name  # type: ignore
        self.setting = PcgSetting(self.stage_name)  # type: ignore
        self.grid = PcgGrid(self.stage_name, self.setting)
        self.agent = Agent(self.grid, self.setting, self.random)
        self.renderer = Renderer(self.grid, self.agent, self.setting)  # type: ignore
        self.info: DefaultDict[str, int] = defaultdict(int)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,))
        self.observation_space = gym.spaces.Box(
            low=0, high=self.setting.DISTANCE_INF, shape=(8,), dtype=np.int32
        )
