import sys
from dataclasses import dataclass

import numpy as np
from ranges import Range
from pytest import approx
from pympler import asizeof
import psutil

from hashlife3d.camera import snapshot_from_grid
from hashlife3d.extent import CuboidExtent, Point2D, RectangleExtent
from hashlife3d.parsers import parse_file
from hashlife3d.grid import Grid, LazyGrid
from hashlife3d.node import Quadtree, QuadtreeBranch, OctreeBranch
from hashlife3d.state import State
from hashlife3d.canonical import _intern_pool


def benchmark_snapshot_from_pattern():
    grid, extent = parse_file(sys.argv[1])
    cuboid = CuboidExtent(
        extent.x_range,
        extent.y_range,
        Range(0, 100),
    )
    snapshot_from_grid(grid, cuboid, Point2D(1, 1))


_process = psutil.Process()

blinker_str = """
____
_X__
_X__
_X__
""".strip()

blinker_grid = LazyGrid(default=State.UNINHABITABLE)
blinker_grid.add_grid(Point2D(0, 0), Grid.from_str(blinker_str))


def main():
    # quadtree = blinker_grid.get_quadtree(RectangleExtent(Range(-1, 7), Range(-2, 6)))
    # octree = quadtree.generate_octree()
    benchmark_snapshot_from_pattern()
    rss = _process.memory_info().rss // 1024
    cache = asizeof.asizeof(_intern_pool) // 1024
    print(f'{rss:,} KB allocated')
    print(f'{cache:,} KB cache')



if __name__ == '__main__':
    main()
