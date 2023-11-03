import sys

import numpy as np
from ranges import Range
from pytest import approx

from hashlife3d.camera import find_required_quadtree, snapshot_from_grid
from hashlife3d.extent import CuboidExtent, RectangleExtent, Point2D
from hashlife3d.parsers.rle import parse as parse_rle
from hashlife3d.grid import Grid, LazyGrid
from hashlife3d.node import QuadtreeBranch


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


def benchmark_octree_from_pattern():
    pattern_file = sys.argv[1]
    lazy_grid = LazyGrid()
    pattern = open(pattern_file, 'r', encoding='utf-8').read()
    grid = parse_rle(pattern)
    lazy_grid.add_grid(Point2D(0, 0), grid)
    cuboid = CuboidExtent(
        Range(0, grid.shape[1]),
        Range(0, grid.shape[0]),
        Range(0, 100),
    )
    (quadtree_extent, octree_extent) = find_required_quadtree(cuboid)
    base_quadtree = lazy_grid.get_quadtree(quadtree_extent)
    octree = base_quadtree.generate_octree()
    print(QuadtreeBranch.generate_octree.cache_stats)


if __name__ == '__main__':
    # benchmark_blinker()
    benchmark_octree_from_pattern()
