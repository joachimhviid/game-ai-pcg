from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    MONSTER_DAMAGE = 5
    POTION_HEAL_AMOUNT = 5
    AGENT_MAX_HEALTH = 10
    