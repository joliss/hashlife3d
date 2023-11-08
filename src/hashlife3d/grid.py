import warnings

import numpy as np
from ranges import Range

from .state import State
from .extent import RectangleExtent, Point2D


def empty_quadtree(extent: RectangleExtent, default=State.DEAD):
    from .node import QuadtreeBranch, QuadtreeLeaf
    assert extent.width == extent.height
    if extent.width == 1:
        return QuadtreeLeaf(default)
    else:
        child = empty_quadtree(RectangleExtent(
            Range(extent.x_range.start, extent.x_range.start + extent.width // 2),
            Range(extent.y_range.start, extent.y_range.start + extent.height // 2)))
        return QuadtreeBranch(np.array([[child, child], [child, child]], dtype=object))


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

    @classmethod
    def dead(cls, width, height):
        grid = np.empty((height, width), dtype=object)
        grid[:] = State.DEAD
        return grid.view(cls)


class LazyGrid:
    def __init__(self, default=State.DEAD):
        assert isinstance(default, State)
        self.default = default
        self._grids = []

    def add_grid(self, offset: Point2D, grid: Grid):
        assert isinstance(grid, Grid)
        assert isinstance(offset, Point2D)
        extent = RectangleExtent(Range(offset.x, offset.x + grid.shape[1]), Range(offset.y, offset.y + grid.shape[0]))
        self._grids.append((extent, grid))

    def get_quadtree(self, extent: RectangleExtent):
        from .node import QuadtreeBranch, QuadtreeLeaf
        assert extent.width == extent.height
        if extent.width == 1:
            return QuadtreeLeaf(self[Point2D(extent.x_range.start, extent.y_range.start)])
        elif not any(extent.intersects(other_extent) for other_extent, _ in self._grids):
            return empty_quadtree(extent, self.default)
        else:
            half_width = extent.width // 2
            half_height = extent.height // 2
            ((nw, ne), (sw, se)) = extent.split_x_y()
            return QuadtreeBranch(np.array([
                [self.get_quadtree(nw), self.get_quadtree(ne)],
                [self.get_quadtree(sw), self.get_quadtree(se)]
            ], dtype=object))

    def __getitem__(self, point: Point2D):
        values = []
        for extent, grid in self._grids:
            if point in extent:
                values.append(grid[point.y - extent.y_range.start, point.x - extent.x_range.start])
        if values:
            if values[0] == State.UNINHABITABLE:
                assert all(value == State.UNINHABITABLE for value in values)
                return State.UNINHABITABLE
            elif any(value == State.ALIVE for value in values): # OR all values
                return State.ALIVE
            else:
                return State.DEAD
        return self.default


class GridFromQuadtree:
    def __init__(self, quadtree):
        self.quadtree = quadtree

    def __getitem__(self, point: Point2D):
        return self.quadtree.get_state(point.x, point.y)

    def initial_quadtree_and_extent(self):
        return self.quadtree, RectangleExtent(
            Range(0, self.quadtree.width()),
            Range(0, self.quadtree.width()),
        )

    def expand_quadtree_and_extent(self, quadtree, extent: RectangleExtent):
        from .node import QuadtreeBranch
        empty = empty_quadtree(extent)
        if (extent.x_range.start + extent.x_range.end) // 2 > 0:
            # Midpoint is positive; expand north-west
            new_extent = RectangleExtent(
                Range(extent.x_range.start - extent.width, extent.x_range.end),
                Range(extent.y_range.start - extent.height, extent.y_range.end),
            )
            new_quadtree = QuadtreeBranch(np.array([
                [empty, empty],
                [empty, quadtree]
            ], dtype=object))
        else:
            # Midpoint is negative; expand south-east
            new_extent = RectangleExtent(
                Range(extent.x_range.start, extent.x_range.end + extent.width),
                Range(extent.y_range.start, extent.y_range.end + extent.height),
            )
            new_quadtree = QuadtreeBranch(np.array([
                [quadtree, empty],
                [empty, empty]
            ], dtype=object))
        return (new_quadtree, new_extent)
