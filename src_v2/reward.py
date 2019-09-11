from enum import Enum

class Reward_Status(Enum):
    NOT_CALCULATED=-2
    UNVISITED=-1
    NOT_AVAILABLE=0
    VISITED=1