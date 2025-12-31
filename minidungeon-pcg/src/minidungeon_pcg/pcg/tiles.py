from enum import StrEnum


class Tiles(StrEnum):
    FLOOR = "."
    WALL = "#"
    START = "S"
    EXIT = "E"
    MONSTER = "M"
    POTION = "P"
    TREASURE = "T"
