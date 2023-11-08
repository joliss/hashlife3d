import numpy as np

from hashlife3d.node import Quadtree, QuadtreeLeaf, QuadtreeBranch, flatten_quadtree_array, flatten_octree_array
from hashlife3d.state import State
from hashlife3d.grid import Grid

blinker_str = """
____
_X__
_X__
_X__
""".strip()

blinker_str2 = """
____
____
XXX_
____
""".strip()


def test_quadtree_string_handling():
    node = Quadtree.from_str(blinker_str)
    assert node.level == 2
    assert str(node) == blinker_str


def test_generate_octree_level_2():
    node = Quadtree.from_str(blinker_str)
    assert str(node.generate_octree()) == """
__
XX
""".strip()


def test_generate_octree_level_3():
    grid = Grid.uninhabitable(8, 8)
    grid[2:6, 2:6] = Grid.from_str(blinker_str)
    node = Quadtree.from_grid(grid)
    octree = node.generate_octree()
    assert str(octree) == '\n\n'.join([blinker_str2, blinker_str])


def test_generate_octree_level_4():
    grid = Grid.uninhabitable(16, 16)
    grid[6:10, 6:10] = Grid.from_str(blinker_str)
    node = Quadtree.from_grid(grid)
    octree = node.generate_octree()
    # Construct expected output by surrounding blinkers with uninhabitable cells
    grid_expected = Grid.uninhabitable(8, 8)
    grid_expected[2:6, 2:6] = Grid.from_str(blinker_str)
    blinker_str_extended = str(grid_expected)
    grid_expected[2:6, 2:6] = Grid.from_str(blinker_str2)
    blinker_str2_extended = str(grid_expected)
    assert str(octree) == '\n\n'.join(
        [blinker_str2_extended, blinker_str_extended] * 2)

    # Test population quadtree
    population_quadtree = octree.quadtrees().population_quadtree()
    assert population_quadtree.to_grid().tolist() == [
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, 0, 0, 0, 0, -1, -1],
        [-1, -1, 0, 2, 0, 0, -1, -1],
        [-1, -1, 2, 4, 2, 0, -1, -1],
        [-1, -1, 0, 2, 0, 0, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1]
    ]


def test_flatten_quadtree_array():
    node = Quadtree.from_str(blinker_str)
    two_by_two = node.children
    four_by_four = flatten_quadtree_array(two_by_two)
    make_leaf = np.vectorize(
        lambda state: QuadtreeLeaf(state), otypes=[object])
    assert np.array_equal(four_by_four, make_leaf(Grid.from_str(blinker_str)))


def test_empty_quadtree():
    for state in (State.DEAD, State.UNINHABITABLE):
        for level in range(4):
            quadtree = Quadtree.empty(level, state)
            assert quadtree.level == level
            grid = quadtree.to_grid()
            assert grid.shape == (2 ** level, 2 ** level)
            assert np.all(grid == state)
