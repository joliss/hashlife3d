import numpy as np
from ranges import Range
from pytest import approx

from hashlife3d.camera import find_required_quadtree, snapshot_from_grid
from hashlife3d.extent import CuboidExtent, RectangleExtent, Point2D
from hashlife3d.grid import Grid, LazyGrid


blinker_str = """
____
_X__
_X__
_X__
""".strip()

blinker_grid = LazyGrid()
blinker_grid.add_grid(Point2D(0, 0), Grid.from_str(blinker_str))


def test_find_required_quadtree():
    assert find_required_quadtree(CuboidExtent(Range(0, 8), Range(0, 8), Range(0, 1))) == (
        RectangleExtent(x_range=Range(-4, 12), y_range=Range(-4, 12)),
        CuboidExtent(x_range=Range(0, 8), y_range=Range(0, 8), t_range=Range(1, 5)),
    )
    assert find_required_quadtree(CuboidExtent(Range(100, 120), Range(-400, -390), Range(0, 1))) == (
        RectangleExtent(x_range=Range(44, 172), y_range=Range(-468, -340)),
        CuboidExtent(x_range=Range(76, 140), y_range=Range(-436, -372), t_range=Range(1, 33)),
)


def test_snapshot_from_grid():
    densities = snapshot_from_grid(blinker_grid, CuboidExtent(Range(0, 3), Range(1, 4), Range(0, 10)), Point2D(3, 3))
    expected = np.array([
        [0.0, 0.5, 0.0],
        [0.5, 1.0, 0.5],
        [0.0, 0.5, 0.0],
    ])
    assert densities == approx(expected)

    densities2 = snapshot_from_grid(blinker_grid, CuboidExtent(Range(0, 3), Range(1, 4), Range(0, 11)), Point2D(3, 3))
    expected2 = np.array([
        [0.0, 6/11, 0.0],
        [5/11, 1.0, 5/11],
        [0.0, 6/11, 0.0],
    ])
    assert densities2 == approx(expected2)
