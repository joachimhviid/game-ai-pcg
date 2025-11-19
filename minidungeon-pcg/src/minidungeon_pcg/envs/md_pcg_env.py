from gym_md.envs.md_env import MdEnvBase


class MdPcgEnv(MdEnvBase):
    def __init__(self, stage_name: str):
        super().__init__(stage_name)
