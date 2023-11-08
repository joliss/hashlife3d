import sys

import numpy as np
from ranges import Range
from pytest import approx

from hashlife3d.camera import snapshot_from_grid
from hashlife3d.extent import CuboidExtent, Point2D
from hashlife3d.parsers import parse_file
from hashlife3d.grid import Grid, LazyGrid


blinker_str = """
____
_X__
_X__
_X__
""".strip()

blinker_grid = LazyGrid()
blinker_grid.add_grid(Point2D(0, 0), Grid.from_str(blinker_str))


def benchmark_blinker():
    densities = snapshot_from_grid(blinker_grid, CuboidExtent(Range(0, 3), Range(1, 4), Range(0, 10**9)), Point2D(3, 3))


def benchmark_snapshot_from_pattern():
    grid, extent = parse_file(sys.argv[1])
    cuboid = CuboidExtent(
        extent.x_range,
        extent.y_range,
        Range(0, 100),
    )
    snapshot_from_grid(grid, cuboid, Point2D(1, 1))


if __name__ == '__main__':
    # benchmark_blinker()
    benchmark_snapshot_from_pattern()
