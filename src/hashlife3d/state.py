from enum import IntEnum


STATE_STRINGS = ['_', 'X', '.']

class State(IntEnum):
    """
    The state of a cell.
    """

    DEAD = 0
    ALIVE = 1
    UNINHABITABLE = 2

    def __str__(self):
        return STATE_STRINGS[self]

    def __repr__(self):
        return f"{self.__class__.__name__}({self})"

    @classmethod
    def from_str(cls, s):
        return cls(STATE_STRINGS.index(s))

    def next_state(self, neighbor_count):
        """
        Implements the standard B3/S23 rule of Conway's Game of Life.
        """
        if self == State.UNINHABITABLE:
            return State.UNINHABITABLE
        elif self == State.ALIVE:
            return State.ALIVE if neighbor_count in (2, 3) else State.DEAD
        else:
            return State.ALIVE if neighbor_count == 3 else State.DEAD
