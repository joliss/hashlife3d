import numpy as np

from hashlife3d.node import QuadTree, QuadTreeLeaf, QuadTreeBranch, flatten_quad_tree_array, flatten_octree_array
from hashlife3d.state import State
from hashlife3d.grid import Grid

blinker_str = """
____
_X__
_X__
_X__
""".strip()

def test_quadtree_string_handling():
    node = QuadTree.from_str(blinker_str)
    assert node.level == 2
    assert str(node) == blinker_str

def test_next_generation_octree_small():
    node = QuadTree.from_str(blinker_str)
    assert str(node.next_generation_octree()) == """
__
XX
""".strip()

def test_next_generation_octree():
    grid = Grid.uninhabitable(8, 8)
    grid[2:6, 2:6] = Grid.from_str(blinker_str)
    node = QuadTree.from_grid(grid)
    octree = node.next_generation_octree()

def test_flatten_quad_tree_array():
    node = QuadTree.from_str(blinker_str)
    two_by_two = node.children
    four_by_four = flatten_quad_tree_array(two_by_two)
    make_leaf = np.vectorize(lambda state: QuadTreeLeaf(state), otypes=[object])
    assert np.array_equal(four_by_four, make_leaf(Grid.from_str(blinker_str)))
