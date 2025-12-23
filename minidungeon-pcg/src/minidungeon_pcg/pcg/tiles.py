from enum import StrEnum


class Tiles(StrEnum):
    WALL = "#"
    FLOOR = "."
    START = "S"
    EXIT = "E"
    MONSTER = "M"
    POTION = "P"
    TREASURE = "T"
