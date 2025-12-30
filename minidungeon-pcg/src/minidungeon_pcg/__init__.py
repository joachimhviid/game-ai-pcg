from gymnasium.envs.registration import register
import sys


stage_name = sys.argv[1] if len(sys.argv) > 1 else "ga_generated"

register(
    id="md-pcg",
    entry_point="minidungeon_pcg.envs:MdPcgEnv",
    kwargs={"stage_name": stage_name},
)

register(
    id="md-pygame",
    entry_point="minidungeon_pcg.envs:MdEnv",
    kwargs={"stage_name": stage_name},
)
