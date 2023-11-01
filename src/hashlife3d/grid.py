import warnings

import numpy as np

from .state import State


class Grid(np.ndarray):
    @classmethod
    def from_list(cls, l):
        return np.asarray(l, dtype=object).view(cls)

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape})"

    def __str__(self):
        if self.ndim != 2:
            warnings.warn(f'Grid should be 2-dimensional, got shape {self.shape}')
            return super().__str__()
        elif self.shape[0] > 0 and self.shape[1] > 0 and not isinstance(self[0, 0], State):
            return super().__str__()
        return '\n'.join(''.join(str(state) for state in row) for row in self)

    @classmethod
    def from_str(cls, s):
        s = s.strip()
        lines = s.splitlines()
        return np.array([[State.from_str(c) for c in line.strip()] for line in lines], dtype=object).view(cls)

    @classmethod
    def uninhabitable(cls, width, height):
        # np.full casts our IntEnum to int64 despite dtype=object, so we first
        # create an empty array and then fill it
        grid = np.empty((height, width), dtype=object)
        grid[:] = State.UNINHABITABLE
        return grid.view(cls)
