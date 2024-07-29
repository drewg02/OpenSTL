from enum import Enum


class ArrayType(str, Enum):
    OUTLINE = 'outline'
    CENTER = 'center'
    PLUS = 'plus'
    RANDOM = 'random'
    RANDOM_UNIFORM = 'random_uniform'
